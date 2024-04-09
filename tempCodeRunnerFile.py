ef print_cube(self):
        mask = (2 ** 3) - 1
        for i in range(len(self.cube)):
            aux = self.cube[i]
            print('FACE', i , '------------')
            for j in range(3):
                for k in range(3):
                    print(aux & mask, end=' ')
                    aux >>= 3
                print()