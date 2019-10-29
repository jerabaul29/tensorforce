from tensorforce.environments import Environment
import socket
from echo_server import EchoServer

from PltDynamicPlot import PltDynamicPlot
from NpRingBuffer import NpRingBuffer
import numpy as np


class RemoteEnvironmentClient(Environment):
    """Used to communicate with a RemoteEnvironmentServer. The idea is that the pair
    (RemoteEnvironmentClient, RemoteEnvironmentServer) allows to transmit information
    through a socket seamlessly.

    The RemoteEnvironmentClient can be directly given to the Runner.

    The RemoteEnvironmentServer herits from a valid Environment add adds the socketing.
    """

    def __init__(self,
                 example_environment,
                 port=12230,
                 host='localhost',
                 verbose=1,
                 buffer_size=262144,
                 ):
        """(port, host) is the necessary info for connecting to the Server socket.
        """

        # templated tensorforce stuff
        self.observation = None
        self.thread = None

        self.buffer_size = buffer_size

        # make arguments available to the class
        # socket
        self.port = port
        self.host = host
        # misc
        self.verbose = verbose
        # states and actions
        self.example_environment = example_environment

        # start the socket
        self.valid_socket = False
        self.socket = socket.socket()
        # if necessary, use the local host
        if self.host is None:
            self.host = socket.gethostname()
        # connect to the socket
        self.socket.connect((self.host, self.port))
        if self.verbose > 0:
            print('Connected to {}:{}'.format(self.host, self.port))
        # now the socket is ok
        self.valid_socket = True

        self.episode = 0
        self.step = 0

        self.perform_plotting = False
    
    def __del__(self):
        if self.valid_socket:
            self.close()        

    def switch_on_action_plotting(self,
                                frequency_plot_execute=1,
                                length_buffers_execute=20):
        
        self.perform_plotting = True

        self.frequency_plot_execute = frequency_plot_execute
        self.length_buffers_execute = length_buffers_execute
        self.n_execute_left_plot = self.frequency_plot_execute
        self.buffer_actions = NpRingBuffer(length=self.length_buffers_execute, shape=(2,))
        self.plot_actions = PltDynamicPlot(min_x=0, max_x=length_buffers_execute, min_y=-5, max_y=5, n_curves=2)

    def states(self):
        return self.example_environment.states()

    def actions(self):
        return self.example_environment.actions()

    def max_episode_timesteps(self):
        return self.example_environment.max_episode_timesteps()

    def close(self):
        to_send = EchoServer.encode_message("CLOSE", 1, verbose=self.verbose)
        self.socket.send(to_send)

    def reset(self):
        # perform the reset
        _ = self.communicate_socket("RESET", 1)

        # get the state
        _, init_state = self.communicate_socket("STATE", 1)

        # Updating episode and step numbers
        self.episode += 1
        self.step = 0

        if self.verbose > 1:
            print("reset done; init_state:")
            print(init_state)

        return(init_state)

    def execute(self, actions):
        if self.perform_plotting:
            self.buffer_actions.push(np.array([actions, 2]))  # 2 here is just to illustrate that can handle 1D action

        # send the control message
        self.communicate_socket("CONTROL", actions)

        # ask to evolve
        self.communicate_socket("EVOLVE", 1)

        # obtain the next state
        _, next_state = self.communicate_socket("STATE", 1)

        # check if terminal
        _, terminal = self.communicate_socket("TERMINAL", 1)

        # get the reward
        _, reward = self.communicate_socket("REWARD", 1)

        # now we have done one more step
        self.step += 1

        if self.perform_plotting:
            if self.n_execute_left_plot <= 0:
                self.n_execute_left_plot = self.frequency_plot_execute

                # plot the action
                number_of_channels = self.buffer_actions.shape[0]
                x = np.arange(0, self.length_buffers_execute, 1)
                x = np.tile(x, (number_of_channels, 1))
                y = self.buffer_actions.get().transpose()
                print(x)
                print(y)
                self.plot_actions.update(x, y)

            self.n_execute_left_plot -= 1

        if self.verbose > 1:
            print("execute performed; state, terminal, reward:")
            print(next_state)
            print(terminal)
            print(reward)

        return (next_state, terminal, reward)

    def communicate_socket(self, request, data):
        """Send a request through the socket, and wait for the answer message.
        """

        to_send = EchoServer.encode_message(request, data, verbose=self.verbose)
        self.socket.send(to_send)

        # TODO: the recv argument gives the max size of the buffer, can be a source of missouts if
        # a message is larger than this; add some checks to verify that no overflow
        received_msg = self.socket.recv(self.buffer_size)

        request, data = EchoServer.decode_message(received_msg, verbose=self.verbose)

        return(request, data)
