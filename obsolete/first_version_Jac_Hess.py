
                    
                elif exc_rank == 2:
                    double_exc_to = [] # [(a1, shift1), (a2, shift2)]
                    double_exc_from = []
                    for i, exc_to in enumerate(det[1:]):
                        if exc_to > self.n_alpha:
                            double_exc_to.append((exc_to, 0 if (i<self.n_alpha) else self.beta_shift))
                    for j in range(self.n_alpha):
                        if j+1 not in det[1: 1 + self.n_alpha]:
                            double_exc_from.append((j+1, 0))
                    for j in range(self.n_alpha):
                        if j+1 not in det[1+self.n_alpha: 1 + 2*self.n_alpha]:
                            double_exc_from.append((j+1, self.beta_shift))
                    if len(double_exc_to) != 2:
                        raise Exception ('len(double_exc_to) != 2' + str(double_exc_to))
                    if len(double_exc_from) != 2:
                        raise Exception ('len(double_exc_from) != 2: ' + str(double_exc_from))

                    exc_from1, exc_from2 = tuple(double_exc_from)
                    exc_to1  , exc_to2   = tuple(double_exc_to)
                    pos1 = get_position_in_jac(exc_from1[0], exc_to1[0], self.n_alpha, exc_from1[1])
                    pos2 = get_position_in_jac(exc_from2[0], exc_to2[0], self.n_alpha, exc_from2[1])
                    if exc_from1[1] == exc_from2[1]: # same spin
                        n_sign = exc_from1[0] - exc_from2[0]
                        if (exc_from1[0] < exc_from2[0] and exc_to1[0] < exc_to2[0]) or\
                           (exc_from1[0] > exc_from2[0] and exc_to1[0] > exc_to2[0]):
                            n_sign += 1
                    else:
                        n_sign = self.n_alpha + self.n_beta + exc_from1[0] - exc_from2[0]
                    if n_sign%2 == 0:
                        Hess[pos1][pos2] += det[0]
                        Hess[pos2][pos1] += det[0]
                    else:
                        Hess[pos1][pos2] -= det[0]
                        Hess[pos2][pos1] -= det[0]

                    if logger.level <= logging.DEBUG:
                        logmsg = []
                        logmsg.append('Double exc: ' + str(det))
                        logmsg.append('double_exc_from: ' + str(double_exc_from))
                        logmsg.append('double_exc_to:   ' + str(double_exc_to))
                        logmsg.append((' K_{{{0:d},{1:d}}}^{{{2:d},{3:d}}}'
                                       + ' spin: {4:s},{5:s} sign={6:s}\n').\
                                      format(double_exc_from[0][0],
                                             double_exc_from[1][0],
                                             double_exc_to[0][0],
                                             double_exc_to[1][0],
                                             'b' if double_exc_from[0][1] else 'a',
                                             'b' if double_exc_from[1][1] else 'a',
                                             '-' if n_sign%2 else '+'))
                        logmsg.append('Added in Hess[{0:d}][{1:d}]'.\
                                      format(pos1,pos2))
                        logger.debug('\n'.join(logmsg))

                    if exc_from1[1] == exc_from2[1]: # same spin
                        exc_to2  , exc_to1 = exc_to1, exc_to2
                        pos1 = _get_position_in_jac(exc_from1[0], exc_to1[0],
                                                    self.n_alpha, exc_from1[1])
                        pos2 = _get_position_in_jac(exc_from2[0], exc_to2[0],
                                                    self.n_alpha, exc_from2[1])
                        if n_sign%2 == 0:
                            Hess[pos1][pos2] -= det[0]
                            Hess[pos2][pos1] -= det[0]
                        else:
                            Hess[pos1][pos2] += det[0]
                            Hess[pos2][pos1] += det[0]
                        if logger.level <= logging.DEBUG:
                            logmsg.append('Added in Hess[{0:d}][{1:d}] with (-)'.\
                                          format(pos1,pos2))
                    if logger.level <= logging.DEBUG:
                        logger.debug('\n'.join(logmsg))
