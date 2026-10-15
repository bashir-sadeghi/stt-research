'''
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 <Your Name or Organization>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-- Bashir Sadeghi
'''

import torch
import pytorch_lightning as pl

import lab.models as models
import lab.losses as losses
import lab.metrics as metrics
import lab.schedulers as schedulers_custom
import pdb

__all__ = ['STT']


class STT(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()

        # Preserve the manual optimization used by the SLM controller.
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['dataloader'])
        self.opts = opts
        self.tokenizer = dataloader.tokenizer
        self.model = getattr(models, opts.model_type)(**opts.model_options)

        # Register separate accumulators for training, validation, and testing.
        self.criterion = torch.nn.ModuleDict({
            f'{stage}_lossc': getattr(losses, opts.loss_type)(**opts.loss_options)
            for stage in ('train', 'val', 'test')
        })
        evaluation_type = getattr(opts, 'evaluation_type_1', 'WordErrorRate')
        evaluation_options = getattr(opts, 'evaluation_options_1', {})
        self.wer = torch.nn.ModuleDict({
            f'{stage}_wer': getattr(metrics, evaluation_type)(**evaluation_options)
            for stage in ('train', 'val', 'test')
        })
        self.cer = torch.nn.ModuleDict({
            f'{stage}_cer': metrics.CharErrorRate()
            for stage in ('train', 'val', 'test')
        })

    def forward(self, features, feature_lengths):
        return self.model(features, feature_lengths)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Only acoustic features need to be on the model's device. The CTC
        # wrapper handles its own target placement; lengths stay available
        # on the CPU for CTC and decoding, including MPS's CPU CTC loss.
        return {
            **batch,
            'features': batch['features'].to(device, non_blocking=True),
            'feature_lengths': batch['feature_lengths'].cpu(),
            'targets': batch['targets'].cpu(),
            'target_lengths': batch['target_lengths'].cpu(),
        }

    def _shared_step(self, batch, stage):
        logits, output_lengths = self.model(
            batch['features'], batch['feature_lengths']
        )
        loss = self.criterion[f'{stage}_lossc'](
            logits,
            output_lengths,
            batch['targets'],
            batch['target_lengths'],
            utterance_ids=batch['ids'],
        )

        # Decoding uses argmax, repetition collapse, and blank removal;
        # it is separate from the differentiable CTC training objective.
        predictions = self.tokenizer.greedy_decode(logits, output_lengths)
        self.wer[f'{stage}_wer'].update(predictions, batch['texts'])
        self.cer[f'{stage}_cer'].update(predictions, batch['texts'])
        return loss, predictions

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        loss, predictions = self._shared_step(batch, 'train')
        if batch_idx == 0:
            self._log_text_samples(batch, predictions, 'train')

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        if self.opts.gradient_clip > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.opts.gradient_clip,
                gradient_clip_algorithm='norm',
            )
        opt.step()
        return loss

    def on_train_epoch_end(self):
        # Match the original controller: step the scheduler once per epoch.
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        self._log_epoch_metrics('train')

    def validation_step(self, batch, batch_idx):
        loss, predictions = self._shared_step(batch, 'val')
        if batch_idx == 0:
            self._log_text_samples(batch, predictions, 'val')
        return loss

    def _log_text_samples(self, batch, predictions, stage):
        # Reuse decoded strings; logging needs no additional model forward pass.
        if (
            self.global_rank != 0
            or self.logger is None
            or not hasattr(self.logger, 'experiment')
            or not hasattr(self.logger.experiment, 'add_text')
        ):
            return

        label = 'Training examples (epoch start)' if stage == 'train' else 'Validation examples'
        blocks = [label]
        for i in range(min(3, len(predictions))):
            blocks.append(
                f"#### Example {i}: {batch['ids'][i]}\n\n"
                f"Reference: {batch['texts'][i]}\n\n"
                f"Prediction: {predictions[i]}"
            )
        self.logger.experiment.add_text(
            f'{stage}_sample',
            '\n\n'.join(blocks),
            global_step=self.current_epoch,
        )

    def on_validation_epoch_end(self):
        self._log_epoch_metrics('val')

    def test_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, 'test')
        return loss

    def on_test_epoch_end(self):
        self._log_epoch_metrics('test')

    def _log_epoch_metrics(self, stage):
        criterion = self.criterion[f'{stage}_lossc']
        wer = self.wer[f'{stage}_wer']
        cer = self.cer[f'{stage}_cer']

        # Logging values are detached by the loss wrapper. The returned batch
        # loss remains connected to the model for self.manual_backward().
        loss = criterion.compute().to(self.device)
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=False)
        self.log(
            f'{stage}_wer', wer.compute(),
            on_epoch=True, prog_bar=(stage == 'val'),
        )
        self.log(f'{stage}_cer', cer.compute(), on_epoch=True, prog_bar=False)

        criterion.reset()
        wer.reset()
        cer.reset()

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate,
            **self.opts.optim_options,
        )

        if self.opts.scheduler_method == 'CustomMultiGammaLR':
            scheduler_gen = schedulers_custom.CustomMultiGammaLR(
                optimizer, **self.opts.scheduler_options
            )
            return [optimizer], [scheduler_gen]
        elif self.opts.scheduler_method is not None:
            scheduler_gen = getattr(
                torch.optim.lr_scheduler, self.opts.scheduler_method
            )(optimizer, **self.opts.scheduler_options)
            return [optimizer], [scheduler_gen]
        else:
            return [optimizer]
