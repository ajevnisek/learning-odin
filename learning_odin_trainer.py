"""Robust Neural Network Trainer."""
import torch
import torch.nn as nn

from classical_trainer import ClassicalTrainer, device
from imagenet_to_imagenet30 import imagenet30_classes_to_imagenet_classes


class LearningODINTrainer(ClassicalTrainer):
    """Train and evaluate Resnet network with Learning ODIN (GQ) loss term."""
    def __init__(self, args):
        super(LearningODINTrainer, self).__init__(args)
        self.lambda_odin = self.arguments.lambda_odin
        self.which_odin_regularization = self.arguments.which_odin_reg
        self.current_test_adv_accuracy = 0

    @staticmethod
    def gradient_quotient_loss_term(inputs, targets, num_of_id_classes,
                                    operands):
        """This method calculates GQ loss term on operands.

        Args:
            inputs: torch.tensor. Input images.
            targets: torch.tensor. Target classes.
            num_of_id_classes: int. |C| = number of classes in the ID train set.
            operands: torch.tensor. logits or softmax scores.

        Returns:
            The Gradient Quotient Loss term.
        """
        odin_goal_nominator = (
                torch.nn.functional.one_hot(targets,
                                            num_of_id_classes) *
                operands).sum()
        grads_nominator = \
            torch.autograd.grad(odin_goal_nominator, inputs,
                                create_graph=True)[0]
        nominator = torch.norm(grads_nominator, p=1)
        odin_goal_denominator = (
                (1 - torch.nn.functional.one_hot(targets, num_of_id_classes))
                * operands).sum()
        grads_denominator = \
            torch.autograd.grad(odin_goal_denominator, inputs,
                                create_graph=True)[0]
        denominator = torch.norm(grads_denominator, p=1)
        gq_loss = nominator / denominator
        return gq_loss

    def train_one_epoch(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(
                self.id_dataset_train_loader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            if self.arguments.in_dataset == 'Imagenet30' and \
                    'Pretrained' in self.arguments.network_name:
                targets = torch.tensor([
                    imagenet30_classes_to_imagenet_classes[x.item()]
                    for x in targets]).to(device)

            inputs.requires_grad = True
            self.optimizer.zero_grad()
            self.net.zero_grad()

            outputs = self.net(inputs)
            # cross entropy loss
            if self.which_odin_regularization == 'one-epoch-gq':
                ce_loss = 0  # fine-tuning is done only with GQ
            else:
                ce_loss = self.criterion(outputs, targets)
            # gradient quotient regularization term
            if self.which_odin_regularization == 'grad-over-grad':
                odin_loss = self.gradient_quotient_loss_term(
                    inputs, targets, self.arguments.id_num_classes, outputs)
            elif self.which_odin_regularization in ['grad-over-grad-to-softmax',
                                                    'one-epoch-gq']:
                s_softmax = nn.Softmax(dim=1)(outputs)
                odin_loss = self.gradient_quotient_loss_term(
                    inputs, targets, self.arguments.id_num_classes, s_softmax)
            else:
                assert False, f"{self.which_odin_regularization} not supported"

            loss = ce_loss + self.lambda_odin * odin_loss
            loss.backward()
            self.optimizer.step()

            inputs.grad = None

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            del loss
            del inputs
            del targets
        nof_batches = len(self.id_dataset_train_loader)
        train_loss = train_loss / nof_batches
        self.print_and_log_train_statistics(epoch, correct, total,
                                            train_loss,
                                            self.current_train_accuracy)

    def run_trainer(self):
        for epoch in range(1, 1 + self.arguments.epochs):
            self.test_one_epoch(epoch)
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)
            self.scheduler.step()
            self.save_checkpoint_if_better(epoch)
            if self.arguments.save_checkpoint_every_epoch:
                self.save_checkpoint_every_epoch(epoch)
