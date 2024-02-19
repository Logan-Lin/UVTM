from torch import nn


class Loss(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = f'LOSS_{name}'


class Encoder(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = f'ENC_{name}'


class Decoder(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = f'DEC_{name}'


class Embed(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = f'EMB_{name}'