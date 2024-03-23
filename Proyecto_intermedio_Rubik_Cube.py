import random

class RubikCube:
    def __init__(self):
        # Lista de listas para representar cada cara del cubo
        self.cube = [
            [[0 for _ in range(3)] for _ in range(3)],  # Cara Frente
            [[1 for _ in range(3)] for _ in range(3)],  # Cara derecha
            [[2 for _ in range(3)] for _ in range(3)],  # Cara superior
            [[3 for _ in range(3)] for _ in range(3)],  # Cara inferior
            [[4 for _ in range(3)] for _ in range(3)],  # Cara izquierda
            [[5 for _ in range(3)] for _ in range(3)]   # Cara trasera
            
        ]
        
        print(len(self.cube))
        print(len(self.cube[0]))
        print(len(self.cube[0][0]))
        #6 en y, 3 en x y z       

    def print_cube(self):
        for cara in self.cube:
            for fila in cara:
                print(fila)
            print()
            
            #BOTTOM DERECHA Y BACK
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

                
    # Gira la columna izquierda en sentido horario
    def move_yleft(self, n_moves):
        pass

    # Gira la columna central en sentido horario
    def move_ymiddle(self, n_moves):#AP
        pass
    # Gira la columna derecha en sentido horario
    def move_yright(self, n_moves):
        pass
    # Gira la cara frontal en sentido horario
    def move_zfront(self, n_moves):#AP
        pass
    # Gira la cara del medio en sentido horario
    def move_zmiddle(self, n_moves):
        pass
    # Gira la cara trasera en sentido horario
    def move_zback(self, n_moves):#YO
        pass        #ACCEDER A LA 0 DE LA 5 

    def __move_cube(self, move):
        if move == 0:
            self.move_xup(1)
        elif move == 1:
            self.move_xbottom(1)
        elif move == 2:
            self.move_xmiddle(3)
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
            self.move_xmiddle(1)
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
                self.print_cube()
        else: 
            for i in range(n_moves):
                move = random.randint(0, 17) #varia el 11
                print(move)
                self.__move_cube(move)

    def solve(self):
        pass

cubo = RubikCube()
cubo.print_cube()
print("-----------------------------------------------")

cubo.move_xbottom(1)
cubo.move_xup(1)
cubo.print_cube()
