import random
from collections import defaultdict
from queue import Queue

class RubikCube:
    def __init__(self):
        # Lista de listas para representar cada cara del cubo
        self.cube = [
            [[0 for _ in range(3)] for _ in range(3)],  # 0 - Cara Frente - BLANCO
            [[1 for _ in range(3)] for _ in range(3)],  # 1 - Cara derecha - ROJO
            [[2 for _ in range(3)] for _ in range(3)],  # 2 - Cara superior - AZUL
            [[3 for _ in range(3)] for _ in range(3)],  # 3 - Cara inferior - VERDE
            [[4 for _ in range(3)] for _ in range(3)],  # 4 - Cara izquierda - NARANJA
            [[5 for _ in range(3)] for _ in range(3)]   # 5 - Cara trasera - AMARILLO
            
        ]
        
    '''print(len(self.cube))
        print(len(self.cube[0]))
        print(len(self.cube[0][0]))'''
        #6 en y, 3 en x y z       

    def print_cube(self):
        for i in range(len(self.cube[0])):
            for cara in self.cube:
                for elem in cara[i]:
                    print(elem, end=' ')
                print('  ', end='') 
            print()  



    # "Y" HACIA ARRIBA
    # 0 - 2 - 5 - 3

    # "X" HACIA ARRIBA
    # 0 - 1 - 5 - 4

    # "Z" HACIA ARRIBA
    # 2 - 1 - 3 - 4


    # Gira la fila superior en sentido horario
    def move_xup(self, n_moves):#AP
        for _ in range(n_moves):
            aux0 = self.cube[0][0].copy()  
            aux1 = self.cube[1][0].copy()  
            aux5 = self.cube[5][0].copy()  
            aux4 = self.cube[4][0].copy()  

            self.cube[1][0] = aux0  
            self.cube[5][0] = aux1  
            self.cube[4][0] = aux5  
            self.cube[0][0] = aux4  
            
    def move_xmiddle(self, n_moves):#C
        self.move_xbottom(n_moves)
        self.move_xup(n_moves)
        
    def move_xbottom(self, n_moves):#AP
        for _ in range(n_moves):
            aux0 = self.cube[0][2].copy()  
            aux1 = self.cube[1][2].copy()  
            aux5 = self.cube[5][2].copy()  
            aux4 = self.cube[4][2].copy()  

            self.cube[1][2] = aux0  
            self.cube[5][2] = aux1  
            self.cube[4][2] = aux5  
            self.cube[0][2] = aux4
        "---------------------------------------------------------------------------------------"

    # Gira la columna izquierda en sentido horario
    def move_yleft(self, n_moves):
        for _ in range(n_moves):
            # Guardar los valores de las posiciones que se moverán en la columna derecha
            aux0 = [self.cube[0][y][0] for y in range(3)] 
            aux2 = [self.cube[2][y][0] for y in range(3)] 
            aux5 = [self.cube[5][y][0] for y in range(3)] 
            aux3 = [self.cube[3][y][0] for y in range(3)] 

            # Mover las posiciones en la columna derecha según el orden 0 - 2 - 5 - 3
            for y in range(3):
                self.cube[2][y][0] = aux0[y]  # de la cara 0 a la cara 2
                self.cube[5][y][0] = aux2[y]  # de la cara 2 a la cara 5
                self.cube[3][y][0] = aux5[y]  # de la cara 5 a la cara 3
                self.cube[0][y][0] = aux3[y]  # de la cara 3 a la cara 0

    def move_ymiddle(self, n_moves):
        self.move_yleft(n_moves)
        self.move_yright(n_moves)

    def move_yright(self, n_moves):
        for _ in range(n_moves):
            aux0 = [self.cube[0][y][2] for y in range(3)]  
            aux2 = [self.cube[2][y][2] for y in range(3)]  
            aux5 = [self.cube[5][y][2] for y in range(3)] 
            aux3 = [self.cube[3][y][2] for y in range(3)] 

            # Mover las posiciones en la columna derecha según el orden 0 - 2 - 5 - 3
            for y in range(3):
                self.cube[2][y][2] = aux0[y]  # de la cara 0 a la cara 2
                self.cube[5][y][2] = aux2[y]  # de la cara 2 a la cara 5
                self.cube[3][y][2] = aux5[y]  # de la cara 5 a la cara 3
                self.cube[0][y][2] = aux3[y]  # de la cara 3 a la cara 0
        "---------------------------------------------------------------------------------------"
    # Gira la cara frontal en sentido horario
    def move_zfront(self, n_moves): # AP
        for _ in range(n_moves):
            # Guardar los valores de las posiciones que se moverán en la columna derecha
            aux0 = [self.cube[2][z][0] for z in range(3)] 
            aux2 = [self.cube[1][z][0] for z in range(3)] 
            aux5 = [self.cube[3][z][0] for z in range(3)] 
            aux3 = [self.cube[4][z][0] for z in range(3)] 

            # Mover las posiciones en la columna derecha según el orden 0 - 2 - 5 - 3
            for z in range(3):
                self.cube[1][z][0] = aux0[z]  # de la cara 2 a la cara 2
                self.cube[3][z][0] = aux2[z]  # de la cara 2 a la cara 5
                self.cube[4][z][0] = aux5[z]  # de la cara 5 a la cara 3
                self.cube[2][z][0] = aux3[z]  # de la cara 3 a la cara 0

    def move_zmiddle(self, n_moves):
        self.move_zback(n_moves)
        self.move_zfront(n_moves)

    # Gira la cara trasera en sentido horario
    def move_zback(self, n_moves): # C
        for _ in range(n_moves):
            aux0 = [self.cube[2][z][2] for z in range(3)] 
            aux2 = [self.cube[1][z][2] for z in range(3)] 
            aux5 = [self.cube[3][z][2] for z in range(3)] 
            aux3 = [self.cube[4][z][2] for z in range(3)] 

            for z in range(3):
                self.cube[1][z][2] = aux0[z]  
                self.cube[3][z][2] = aux2[z]  
                self.cube[4][z][2] = aux5[z] 
                self.cube[2][z][2] = aux3[z]  

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
        elif move == 9:
            self.move_xup(3)
        elif move == 10:
            self.move_xbottom(3)
        elif move == 11:
            self.move_xmiddle(3)
        elif move == 12:
            self.move_yleft(3)
        elif move == 13:
            self.move_ymiddle(3)
        elif move == 14:
            self.move_yright(3)
        elif move == 15:
            self.move_zback(3)
        elif move == 16:
            self.move_zfront(3)
        elif move == 17:
            self.move_zmiddle(3)

    def shuffle(self, n_moves, make_moves=[]):
        if len(make_moves) > 0:
            for move in make_moves:
                self.__move_cube(move)
                
        else: 
            for i in range(n_moves):
                move = random.randint(0, 17) #varia el 17
                self.__move_cube(move)

   # PARALELOOOOOOOOOOOOOOO, METER HILO
                
class RubikSolver(RubikCube):
    def __init__(self):
        super().__init__()


    def breath_first_search(self, cube):
        pass

    def best_first_search(self, cube):
        pass

    def a_star(self, cube):
        pass

    def new_method(self, cube):
        pass

cubo = RubikSolver()

print("\n*******Movimientos que puedes hacer*******\n")
print("Mover fila superior a la derecha(0) o izquierda(9)\nMover fila intermedia a la derecha(2) o izquierda(11)")
print("Mover fila inferior a la derecha(1) o izquierda(10)")
print("Mover columna izquierda hacia arriba(3) o hacia abajo(12)\nMover columna intermedia hacia arriba(4) o hacia bajo(13)")
print("Mover columna derecha hacia arriba(5) o hacia abajo(14)")
print("Mover cara frontal a la derecha(7) o izquierda(16)\nMover cara intermedia a la derecha(8) o izquierda(17)")
print("Mover cara trasera a la derecha(6) o izquierda(15)")
print("***********************************************************\n")
print("Cubo original:")
cubo.print_cube()
print("---------------------------------------------------------")
print("\nCubo revuelto al azar:")
cubo.shuffle(1)
cubo.print_cube()
print("---------------------------------------------------------")
print("\nCubo revuelto manualmente:")
movimientos_manual = [5]
cubo.shuffle( 1, movimientos_manual)
cubo.print_cube()
print("---------------------------------------------------------")

