# written by David Sommer (david.sommer at inf.ethz.ch)

import io
import json
from time import sleep
from threading import Thread
from contextlib import redirect_stdout
from channels.generic.websocket import WebsocketConsumer


import sys
sys.path.insert(0, "../")
from interfaces import executeGaussian, executeLaplace, executeHistogram


class Consumer(WebsocketConsumer):

    distr_1 = None
    distr_2 = None
    parameters = None

    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        print("DISONNECT")

    def abort(self):
        print("Abort")
        self.send_console_output("ERROR: reload browser tab!")
        self.close()

    def receive(self, text_data):
        print(text_data)

        # query_details = self.parse_query_string(text_data)
        f = io.StringIO()
        with redirect_stdout(f):
            ready = self.parse_parameters(text_data)

            if ready:
                # thread = Thread(target = compute_request, kwargs = {'parameters': self.parameters, 'distr_1': self.distr_1, 'distr_2': self.distr_2})
                thread = ComputationThread(parameters = self.parameters, distr_1 = self.distr_1, distr_2 = self.distr_2)
                thread.start()

                pos = 0
                while thread.is_alive():
                    f.seek(pos)
                    output = f.read()
                    pos += len(output)

                    self.send_console_output(output)
                    sleep(1)
                thread.join()  # should be finished.. right?

                # sending leftover console output
                f.seek(pos)
                output = f.read()
                self.send_console_output(output)

                # sending images
                image_answer = {'type': 'image_1', 'data': thread.image_1 }
                self.send(json.dumps(image_answer))

                # we are done
                self.close()

    # parses parameters and decides if all required parameters have been received.
    # returns true if computation is ready to start.
    def parse_parameters(self, text_string):
        try:
            msg = json.loads(text_string)
            msg_type = msg['type']
            msg_data = msg['data']

            # parse message type
            if msg_type == 'parameters':
                self.parameters = msg_data
            elif msg_type == 'file':
                if msg_data['name'] == 'distr_1':
                    self.distr_1 = msg_data['data']
                    print("HERE1")
                elif msg_data['name'] == 'distr_2':
                    print("HERE2")
                    self.distr_2 = msg_data['data']
                else:
                    raise Exception("Unknown file received")
            else:
                raise Exception("Unknown request format")

            # check if ready
            if self.parameters['type'] == 'Custom':
                if not ( self.distr_1 and self.distr_2 ):
                    return False
            return True

        except Exception as e:
            print("Error: " + str(e))
            self.abort()

    def send_console_output(self, output):
        msg = {'type': 'console', 'data': output}
        self.send(json.dumps(msg))


class ComputationThread(Thread):

    image_1 = None

    def __init__(self, parameters, distr_1, distr_2):
        self.parameters = parameters
        self.distr_1 = distr_1
        self.distr_2 = distr_2

        super(ComputationThread, self).__init__()

    def run(self):

        da_type = self.parameters['type']
        if da_type == 'Gaussian':
            n_gaussian = int(self.parameters['n_gauss'])
            sigma = int(self.parameters['sigma'])
            print("Gaussian",sigma, n_gaussian)
            self.image_1 = executeGaussian(sigma, n_gaussian)
        elif da_type == 'Laplace':
            n_lap = int(self.parameters['n_lap'])
            scale = int(self.parameters['scale'])
            print("laplace",scale, n_lap)
            self.image_1 = executeLaplace(scale, n_lap)
        elif da_type == 'Custom':
            print("custom")
            n_cust = int(self.parameters['n_cust'])
            print("custom distr_1 distr_2")
            self.image_1 = executeHistogram(self.distr_1, self.distr_2, n_cust)
        else:
            raise Exception('Invalid type!')
