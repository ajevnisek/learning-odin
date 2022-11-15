"""Train OOD Detection via In-Distribution Robustness."""
from parse_cmd_args import parse_arguments
from classical_trainer import ClassicalTrainer
from learning_odin_trainer import LearningODINTrainer


def main():
    args = parse_arguments()
    if args.which_robust_optimization == 'Classical':
        trainer = ClassicalTrainer(args)
    elif args.which_robust_optimization == 'Learning-ODIN-Optimization':
        trainer = LearningODINTrainer(args)
    else:
        assert False, f'{args.which_robust_optimization} optimization not' \
                      f' supported'
    trainer.run_trainer()


if __name__ == "__main__":
    main()
