# Description: Main file to run the training of the model

from src import training_model, sys_config, hand_write_gui
from src.hand_write_gui import Gui

if __name__ == '__main__':
    # train the model
    # training_model.training_model()

    # run the gui
    app = Gui()
    app.mainloop()

    # check the system configuration
    # sys_config.check_config()
