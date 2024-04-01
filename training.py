from model import MachineTranslation

def model_training(mt,name):
    mt.build_model()
    mt.train()
    mt.save_model(name)



def main():
    mt = MachineTranslation()
    mt.load_data(r"D:\Datasets\spa.txt")
    model_training(mt,"testing.h5")

main()