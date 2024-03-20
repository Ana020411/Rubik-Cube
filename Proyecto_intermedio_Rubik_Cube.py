import random

class RubikCube:
    def __init__(self):
        # Lista de listas para representar cada cara del cubo
        self.cube = [
            [[0 for _ in range(3)] for _ in range(3)],  # Cara superior
            [[1 for _ in range(3)] for _ in range(3)],  # Cara inferior
            [[2 for _ in range(3)] for _ in range(3)],  # Cara izquierda
            [[3 for _ in range(3)] for _ in range(3)],  # Cara derecha
            [[4 for _ in range(3)] for _ in range(3)],  # Cara frontal
            [[5 for _ in range(3)] for _ in range(3)]   # Cara trasera
        ]

    def print_cube(self):
        for cara in self.cube:
            for fila in cara:
                print(fila)
            print()
    
    def __xmove(self, n_row, n_moves):
        self.cube[n_row] = [self.cube[n_row][(i - n_moves) % 3] for i in range(3)]

    def __ymove(self, n_col, n_moves):
        for i in range(3):
            self.cube[i][n_col] = self.cube[(i - n_moves) % 3][n_col]

    def __zmove(self, n_depth, n_moves):
        for i in range(3):
            self.cube[i] = [self.cube[i][j] for j in range(n_moves, 3)] + [self.cube[i][j] for j in range(n_moves)]

    # Gira la fila superior en sentido horario
    def move_xup(self, n_moves):
        self.__xmove(0, n_moves)

    # Gira la fila central en sentido horario
    def move_xmiddle(self, n_moves):
        self.__xmove(1, n_moves)

    # Gira la fila inferior en sentido horario
    def move_xbottom(self, n_moves):
        self.__xmove(2, n_moves)

    # Gira la columna izquierda en sentido horario
    def move_yleft(self, n_moves):
        self.__ymove(0, n_moves)

    # Gira la columna central en sentido horario
    def move_ymiddle(self, n_moves):
        self.__ymove(1, n_moves)

    # Gira la columna derecha en sentido horario
    def move_yright(self, n_moves):
        self.__ymove(2, n_moves)

    # Gira la cara frontal en sentido horario
    def move_zfront(self, n_moves):
        self.__zmove(0, n_moves)

    # Gira la cara del medio en sentido horario
    def move_zmiddle(self, n_moves):
        self.__zmove(1, n_moves)

    # Gira la cara trasera en sentido horario
    def move_zback(self, n_moves):
        self.__zmove(2, n_moves)

    def __move_cube(self, move):
        if move == 0:
            self.move_xup(1)
        elif move == 1:
            self.move_xbottom(1)
        elif move == 2:
            self.move_xmiddle(1)
        elif move == 3:
            self.move_yleft(1)
        elif move == 4:
            self.move_ymiddle(1)
        elif move == 5:
            self.move_yright(1)
        elif move == 6:
            self.move_zback(1)
        elif move == 7:
            self.move_zfront(1)
        elif move == 8:
            self.move_zmiddle(1)

    def __count_colors(self):
        pass

    def shuffle(self, n):
        movimientos_posibles = [self.move_xup, self.move_xmiddle, self.move_xbottom, self.move_yleft, self.move_ymiddle, self.move_yright, self.move_zfront, self.move_zmiddle, self.move_zback]

        for _ in range(n):
            movimiento_aleatorio = random.choice(movimientos_posibles)
            movimiento_aleatorio(1)

    def solve(self):
        pass

cubo = RubikCube()
cubo.print_cube()
cubo.shuffle(10)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
cubo.print_cube()