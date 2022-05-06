class solver:

    def __init__(self, data_init, description):

        import numpy as np
        self.data_init = np.copy(data_init)
        self.desc = description

    def solve(self, border_type='tor', deg=1, vis=False):

        import numpy as np
        from tqdm.notebook import tqdm

        self.data_current = np.copy(self.data_init)
        self.data_next = np.zeros_like(self.data_current)

        if vis:
            r = tqdm(range(int(self.desc['T'] / self.desc['dt'])))
        else:
            r = range(int(self.desc['T'] / self.desc['dt']))

        for k in r:

            for j in range(self.desc['ny']):

                for i in range(self.desc['nx']):

                    w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, i])
                    if border_type == 'tor':
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, (i - 1) % self.desc['nx']])
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, (i + 1) % self.desc['nx']])
                    elif border_type == 'absorb':
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, (i - 1) * (i > 0)])
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, (i + 1) * (i < (self.desc['nx'] - 1)) + (self.desc['nx'] - 1) * (i >= (self.desc['nx'] - 1))])

                    if deg == 1:
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p)
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2)
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p)
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4)
                        w5_new = w5
                    elif deg == 2:
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / (2 * self.desc['h']) * (w1n - w1p) + (self.desc['cp'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w1n - 2 * w1 + w1p)
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / (2 * self.desc['h']) * (w2n - w2p) + (self.desc['cp'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w2n - 2 * w2 + w2p)
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / (2 * self.desc['h']) * (w3n - w3p) + (self.desc['cs'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w3n - 2 * w3 + w3p)
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / (2 * self.desc['h']) * (w4n - w4p) + (self.desc['cs'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w4n - 2 * w4 + w4p)
                        w5_new = w5

                    self.data_next[j, i]['vx'], \
                    self.data_next[j, i]['vy'], \
                    self.data_next[j, i]['sigxx'], \
                    self.data_next[j, i]['sigxy'], \
                    self.data_next[j, i]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new)

            for j in range(self.desc['ny']):

                for i in range(self.desc['nx']):

                    w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[j, i])
                    if border_type == 'tor':
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[(j - 1) % self.desc['ny'], i])
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[(j + 1) % self.desc['ny'], i])
                    elif border_type =='absorb':
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[(j - 1) * (j > 0), i])
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[(j + 1) * (j < (self.desc['ny'] - 1)) + (self.desc['ny'] - 1) * (j >= (self.desc['ny'] - 1)), i])

                    if deg == 1:
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p)
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2)
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p)
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4)
                        w5_new = w5
                    elif deg == 2:
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / (2 * self.desc['h']) * (w1n - w1p) + (self.desc['cp'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w1n - 2 * w1 + w1p)
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / (2 * self.desc['h']) * (w2n - w2p) + (self.desc['cp'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w2n - 2 * w2 + w2p)
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / (2 * self.desc['h']) * (w3n - w3p) + (self.desc['cs'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w3n - 2 * w3 + w3p)
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / (2 * self.desc['h']) * (w4n - w4p) + (self.desc['cs'] * self.desc['dt']) ** 2 / (2 * self.desc['h'] ** 2) * (w4n - 2 * w4 + w4p)
                        w5_new = w5

                    self.data_current[j, i]['vx'], \
                    self.data_current[j, i]['vy'], \
                    self.data_current[j, i]['sigxx'], \
                    self.data_current[j, i]['sigxy'], \
                    self.data_current[j, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new)

        return self.data_current
    
    def solve_free(self, deg=1, vis=False):

        import numpy as np
        from tqdm.notebook import tqdm

        self.data_current = np.copy(self.data_init)
        self.data_next = np.zeros_like(self.data_current)

        if vis:
            r = tqdm(range(int(self.desc['T'] / self.desc['dt'])))
        else:
            r = range(int(self.desc['T'] / self.desc['dt']))

        if deg == 1:

            for k in r: 

                for j in range(self.desc['ny']): 
                    w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, self.desc['nx'] - 1])
                    w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, self.desc['nx'] - 2])
                    w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p)
                    w2_new = - w1_new
                    w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p) 
                    w4_new = - w3_new 
                    w5_new = w5 
                    self.data_next[j, self.desc['nx']-1]['vx'], \
                    self.data_next[j, self.desc['nx']-1]['vy'], \
                    self.data_next[j, self.desc['nx']-1]['sigxx'], \
                    self.data_next[j, self.desc['nx']-1]['sigxy'], \
                    self.data_next[j, self.desc['nx']-1]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                    w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, 0])
                    w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, 1])
                    w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2) 
                    w1_new = - w2_new 
                    w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4) 
                    w3_new = - w4_new 
                    w5_new = w5 
                    self.data_next[j, 0]['vx'], \
                    self.data_next[j, 0]['vy'], \
                    self.data_next[j, 0]['sigxx'], \
                    self.data_next[j, 0]['sigxy'], \
                    self.data_next[j, 0]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                    for i in range(1, self.desc['nx']-1): 
                        w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, i]) 
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, i - 1]) 
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, i + 1]) 
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p) 
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2) 
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p) 
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4) 
                        w5_new = w5 
                        self.data_next[j, i]['vx'], \
                        self.data_next[j, i]['vy'], \
                        self.data_next[j, i]['sigxx'], \
                        self.data_next[j, i]['sigxy'], \
                        self.data_next[j, i]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 

                for j in range(self.desc['ny']): 
                    for i in range(self.desc['nx']): 
                        if j == self.desc['ny'] - 1: 
                            w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[self.desc['ny']-1, i]) 
                            w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[self.desc['ny']-2, i]) 
                            w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p) 
                            w2_new = - w1_new 
                            w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p) 
                            w4_new = - w3_new 
                            w5_new = w5 
                            self.data_current[self.desc['ny']-1, i]['vx'], \
                            self.data_current[self.desc['ny']-1, i]['vy'], \
                            self.data_current[self.desc['ny']-1, i]['sigxx'], \
                            self.data_current[self.desc['ny']-1, i]['sigxy'], \
                            self.data_current[self.desc['ny']-1, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                        elif j == 0: 
                            w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[0, i]) 
                            w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[1, i]) 
                            w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2) 
                            w1_new = - w2_new 
                            w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4) 
                            w3_new = - w4_new 
                            w5_new = w5 
                            self.data_current[0, i]['vx'], \
                            self.data_current[0, i]['vy'], \
                            self.data_current[0, i]['sigxx'], \
                            self.data_current[0, i]['sigxy'], \
                            self.data_current[0, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                        else: 
                            w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[j, i]) 
                            w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[j - 1, i]) 
                            w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[j + 1, i]) 
                            w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w1 - w1p) 
                            w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * (w2n - w2) 
                            w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w3 - w3p) 
                            w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * (w4n - w4) 
                            w5_new = w5 
                            self.data_current[j, i]['vx'], \
                            self.data_current[j, i]['vy'], \
                            self.data_current[j, i]['sigxx'], \
                            self.data_current[j, i]['sigxy'], \
                            self.data_current[j, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new)

        if deg == 2:

            for k in r: 

                for j in range(self.desc['ny']):

                    w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, self.desc['nx'] - 1])
                    w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, self.desc['nx'] - 2])
                    w1pp, w2pp, w3pp, w4pp, w5pp = self.__OmegaX__(self.data_current[j, self.desc['nx'] - 3])
                    w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w1 - 4 * w1p + w1pp) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w1 - 2 * w1p + w1pp) 
                    w2_new = - w1_new 
                    w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w3 - 4 * w3p + w3pp) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w3 - 2 * w3p + w3pp) 
                    w4_new = - w3_new 
                    w5_new = w5
                    self.data_next[j, self.desc['nx'] - 1]['vx'], \
                    self.data_next[j, self.desc['nx'] - 1]['vy'], \
                    self.data_next[j, self.desc['nx'] - 1]['sigxx'], \
                    self.data_next[j, self.desc['nx'] - 1]['sigxy'], \
                    self.data_next[j, self.desc['nx'] - 1]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                    w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, 0]) 
                    w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, 1]) 
                    w1nn, w2nn, w3nn, w4nn, w5nn = self.__OmegaX__(self.data_current[j, 2]) 
                    w2_new = w2 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w2 - 4 * w2n + w2nn) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w2 - 2 * w2n + w2nn) 
                    w1_new = - w2_new 
                    w4_new = w4 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w4 - 4 * w4n + w4nn) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w4 - 2 * w4n + w4nn) 
                    w3_new = - w4_new 
                    w5_new = w5 
                    self.data_next[j, 0]['vx'], \
                    self.data_next[j, 0]['vy'], \
                    self.data_next[j, 0]['sigxx'], \
                    self.data_next[j, 0]['sigxy'], \
                    self.data_next[j, 0]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                    
                    for i in range(1, self.desc['nx'] - 1): 

                        w1, w2, w3, w4, w5 = self.__OmegaX__(self.data_current[j, i]) 
                        w1p, w2p, w3p, w4p, w5p = self.__OmegaX__(self.data_current[j, i - 1]) 
                        w1n, w2n, w3n, w4n, w5n = self.__OmegaX__(self.data_current[j, i + 1]) 
                        w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w1n - w1p) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w1n + w1p - 2 * w1) 
                        w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w2n - w2p) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w2n + w2p - 2 * w2) 
                        w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w3n - w3p) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w3n + w3p - 2 * w3) 
                        w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w4n - w4p) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w4n + w4p - 2 * w4) 
                        w5_new = w5 
                        self.data_next[j, i]['vx'], \
                        self.data_next[j, i]['vy'], \
                        self.data_next[j, i]['sigxx'], \
                        self.data_next[j, i]['sigxy'], \
                        self.data_next[j, i]['sigyy'] = self.__OmegaXInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                
                for j in range(self.desc['ny']): 
                    
                        for i in range(self.desc['nx']): 
                        
                            if j == self.desc['ny'] - 1: 
                                w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[self.desc['ny'] - 1, i]) 
                                w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[self.desc['ny'] - 2, i]) 
                                w1pp, w2pp, w3pp, w4pp, w5pp = self.__OmegaY__(self.data_next[self.desc['ny'] - 3, i]) 
                                w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w1 - 4 * w1p + w1pp) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w1 - 2 * w1p + w1pp) 
                                w2_new = - w1_new 
                                w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w3 - 4 * w3p + w3pp) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w3 - 2 * w3p + w3pp) 
                                w4_new = - w3_new 
                                w5_new = w5 
                                self.data_current[self.desc['ny'] - 1, i]['vx'], \
                                self.data_current[self.desc['ny'] - 1, i]['vy'], \
                                self.data_current[self.desc['ny'] - 1, i]['sigxx'], \
                                self.data_current[self.desc['ny'] - 1, i]['sigxy'], \
                                self.data_current[self.desc['ny'] - 1, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                            elif j == 0: 
                                w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[0, i]) 
                                w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[1, i]) 
                                w1nn, w2nn, w3nn, w4nn, w5nn = self.__OmegaY__(self.data_next[2, i]) 
                                w2_new = w2 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w2 - 4 * w2n + w2nn) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w2 - 2 * w2n + w2nn) 
                                w1_new = - w2_new 
                                w4_new = w4 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (3 * w4 - 4 * w4n + w4nn) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w4 - 2 * w4n + w4nn) 
                                w3_new = - w4_new 
                                w5_new = w5 
                                self.data_current[0, i]['vx'], \
                                self.data_current[0, i]['vy'], \
                                self.data_current[0, i]['sigxx'], \
                                self.data_current[0, i]['sigxy'], \
                                self.data_current[0, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new) 
                            else: 
                                w1, w2, w3, w4, w5 = self.__OmegaY__(self.data_next[j, i]) 
                                w1p, w2p, w3p, w4p, w5p = self.__OmegaY__(self.data_next[j - 1, i]) 
                                w1n, w2n, w3n, w4n, w5n = self.__OmegaY__(self.data_next[j + 1, i]) 
                                w1_new = w1 - self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w1n - w1p) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w1n + w1p - 2 * w1) 
                                w2_new = w2 + self.desc['cp'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w2n - w2p) + (self.desc['cp'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w2n + w2p - 2 * w2) 
                                w3_new = w3 - self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w3n - w3p) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w3n + w3p - 2 * w3) 
                                w4_new = w4 + self.desc['cs'] * self.desc['dt'] / self.desc['h'] * 0.5 * (w4n - w4p) + (self.desc['cs'] * self.desc['dt'] / self.desc['h']) ** 2 * 0.5 * (w4n + w4p - 2 * w4) 
                                w5_new = w5 
                                self.data_current[j, i]['vx'], \
                                self.data_current[j, i]['vy'], \
                                self.data_current[j, i]['sigxx'], \
                                self.data_current[j, i]['sigxy'], \
                                self.data_current[j, i]['sigyy'] = self.__OmegaYInv__(w1_new, w2_new, w3_new, w4_new, w5_new)

        return self.data_current

    def __OmegaX__(self, q):

        import numpy as np

        vx, vy, sigxx, sigxy, sigyy = q

        return (self.desc['rho'] * (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) * vx / (2.0 * self.desc['cp']) + (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) * sigxx / (2.0 * self.desc['cp'] ** 2),
            -self.desc['rho'] * (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) * vx / (2.0 * self.desc['cp']) + (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) * sigxx / (2.0 * self.desc['cp'] ** 2),
            self.desc['rho'] * self.desc['cs'] * vy / 2.0 + sigxy / 2.0,
            -self.desc['rho'] * self.desc['cs'] * vy / 2.0 + sigxy / 2.0,
            -sigxx + (2 * self.desc['cs'] ** 2) / (self.desc['cp'] ** 2) * sigxx + sigyy)

    def __OmegaXInv__(self, w1, w2, w3, w4, w5):

        import numpy as np
        
        return (self.desc['cp'] / (self.desc['rho'] * self.desc['cp'] ** 2 - 2.0 * self.desc['rho'] * self.desc['cs'] ** 2) * w1 - self.desc['cp'] / (self.desc['rho'] * self.desc['cp'] ** 2 - 2.0 * self.desc['rho'] * self.desc['cs'] ** 2) * w2,
            1 / (self.desc['rho'] * self.desc['cs']) * w3 - 1 / (self.desc['rho'] * self.desc['cs']) * w4,
            (self.desc['cp'] ** 2) / (self.desc['cp'] ** 2 - 2.0 * self.desc['cs'] ** 2) * w1 + (self.desc['cp'] ** 2) / (self.desc['cp'] ** 2 - 2.0 * self.desc['cs'] ** 2) * w2,
            w3 + w4,
            w1 + w2 + w5)

    def __OmegaY__(self, q):

        import numpy as np
        
        vx, vy, sigxx, sigxy, sigyy = q
        
        return (self.desc['rho'] * self.desc['cp'] * vy / 2.0 + sigyy / 2.0,
            -self.desc['rho'] * self.desc['cp'] * vy / 2.0 + sigyy / 2.0,
            self.desc['rho'] * self.desc['cs'] * vx / 2.0 + sigxy / 2.0,
            -self.desc['rho'] * self.desc['cs'] * vx / 2.0 + sigxy / 2.0,
            sigxx - sigyy + (2 * self.desc['cs'] ** 2) / (self.desc['cp'] ** 2) * sigyy)

    def __OmegaYInv__(self, w1, w2, w3, w4, w5):

        import numpy as np
        
        return (1 / (self.desc['rho'] * self.desc['cs']) * w3 - 1 / (self.desc['rho'] * self.desc['cs']) * w4,
            1 / (self.desc['rho'] * self.desc['cp']) * w1 - 1 / (self.desc['rho'] * self.desc['cp']) * w2,
            (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) / (self.desc['cp'] ** 2) * w1 + (self.desc['cp'] ** 2 - 2 * self.desc['cs'] ** 2) / (self.desc['cp'] ** 2) * w2 + w5,
            w3 + w4,
            w1 + w2)