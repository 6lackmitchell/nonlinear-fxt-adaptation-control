"""recurrent_neural_network.py

Implements a class interfaces to different Recurrent Neural Networks.

Classes:
    RecurrentNeuralNetwork

"""
from nptyping import NDArray
import numpy as np


def sigmoid(x: float or NDArray):
    """Numerically stable implementation of the sigmoid function.

    Arguments:
        x: input

    Returns:
        sigmoid(x)

    """
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))

    return sig


class RecurrentNeuralNetwork:
    """RecurrentNeuralNetwork: class interface to recurrent neural network.

    Properties:
        TBD

    Methods:
        TBD

    """

    def __init__(self, n_inputs: int, n_outputs: int):
        """Class initializer.

        Arguments:
            n_inputs: number of RNN inputs
            n_outputs: number of RNN outputs

        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Weights
        self.input_weights_a = 0.1 * np.eye(n_inputs)  # Forget Weights
        self.input_weights_b = np.eye(n_inputs)
        self.input_weights_c = np.eye(n_inputs)
        self.input_weights_d = np.eye(n_inputs)
        self.hidden_weights_a = 0.1 * np.eye(n_outputs)  # Forget Weights
        self.hidden_weights_b = np.eye(n_outputs)
        self.hidden_weights_c = np.eye(n_outputs)
        self.hidden_weights_d = np.eye(n_outputs)

        # Biases
        self.bias_a = np.random.random((n_inputs,))
        self.bias_b = np.random.random((n_inputs,))
        self.bias_c = np.random.random((n_inputs,))
        self.bias_d = np.random.random((n_inputs,))

        # RNN States
        self.cell_state = np.zeros((self.n_inputs,))
        self.hidden_state = np.zeros((self.n_outputs,))

    #! This is really for a LSTM RNN, will generalize later
    def update_rnn(self, new_input: NDArray) -> NDArray:
        """Updates the RNN state according to the new input.

        Arguments:
            new_input: new input to the RNN

        Returns:
            new_output: new output based on input, hidden, and cell states

        """
        at = sigmoid(
            self.hidden_weights_a @ self.hidden_state
            + self.input_weights_a @ new_input
            + self.bias_a
        )
        bt = sigmoid(
            self.hidden_weights_b @ self.hidden_state
            + self.input_weights_b @ new_input
            + self.bias_b
        )
        ct = np.tanh(
            self.hidden_weights_c @ self.hidden_state
            + self.input_weights_c @ new_input
            + self.bias_c
        )
        dt = sigmoid(
            self.hidden_weights_d @ self.hidden_state
            + self.input_weights_d @ new_input
            + self.bias_d
        )

        self.cell_state = self.hidden_state * at + bt * ct
        self.hidden_state = dt * np.tanh(self.cell_state)

        return self.hidden_state

    @property
    def outputs(self) -> NDArray:
        """Property for the hidden RNN states."""
        return self.hidden_state
