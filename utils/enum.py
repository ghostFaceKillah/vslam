from enum import Enum


class StrEnum(str, Enum):
    ...



class Example(StrEnum):
    OKE = 'oke'
    WER = 'wer'



class A:
    def __init__(self, *args):
        x = 1

if __name__ == '__main__':
    # print(Example.WER)
    # print(f"'wer' == Example.WER => {Example.WER == 'wer'}")
    # print(f"Example.WER == Example.WER => {Example.WER == Example.WER}")
    # print(f"Example.WER == Example.OKE => {Example.WER == Example.OKE}")
    a = A(1, 2,3, 4)
