from model import MachineTranslation


def main():
    mt = MachineTranslation()
    mt.load_data(r"D:\Datasets\spa.txt")
    mt.model_load(r"D:\vs code\python\DeepLearning\Projects\eng_fre_translation\eng_fre5.h5")
    mt.build_inference_model()

    for seq_index in range(50):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = mt.encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = mt.decode_sequence(input_seq)
        print('-')
        print('Input sentence:', mt.input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

main()