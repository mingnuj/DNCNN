from trainer.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("config/train_config.yaml")
    trainer.train(continue_train=False)
