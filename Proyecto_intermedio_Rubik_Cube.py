import random
from collections import deque
from queue import Queue
import copy
from queue import PriorityQueue

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
        return self

   # PARALELOOOOOOOOOOOOOOO, METER HILO
#----------------------------------------------------CLASE HEURISTICA---------------------------------------
class Heuristica:
    @staticmethod
    def bfs(node_a, node_b):
        cubo_uno = node_a.state.cube
        cubo_dos = node_b.state.cube
        distancia = 0

        for face in range(6):
            for row in range(3):
                for col in range(3):
                    valor_cubouno = cubo_uno[face][row][col]
                    face_target, row_target, col_target = -1, -1, -1
                    for c in range(6):
                        for f in range(3):
                            for col in range(3):
                                if cubo_dos[c][f][col] == valor_cubouno:
                                    face_target, row_target, col_target = c, f, col
                                    break
                        if face_target != -1:
                            break
                    distancia += abs(face - face_target) + abs(row - row_target) + abs(col - col_target)#CHECAAAAAAAR SI ACUMULAMOS LAS DISTANCIAS O NO*****

        return distancia
    
    @staticmethod
    def heuristic2(node_a, node_b):
        pass
    
#-----------------------------------------------------CLASE NODO---------------------------------------
class Node:
    def __init__(self, state,  path=None):
        self.state = state
        self.heuristic_value = -1
        self.path = []
        self.path = [] if path is None else path

    def calculate_heuristic(self, other, heuristic):
        self.heuristic_value = heuristic(self, other)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.state == other.state

    def __lt__(self, other):
        if not isinstance(other, Node):
            return False
        return self.heuristic_value < other.heuristic_value

    def __gt__(self, other):
        if not isinstance(other, Node):
            return False
        return self.heuristic_value > other.heuristic_value
    
#-----------------------------------------------------CLASE NODO A*---------------------------------------
    
class NodeAStar(Node):
    def __init__(self, state,  path=None):
        super().__init__(state,  path=path)
        self.distance = 0

    def __lt__(self, other):
        if not isinstance(other, NodeAStar):
            return False
        return (self.distance + self.heuristic_value) < (other.distance + other.heuristic_value)
    
    def __gt__(self, other):
        if not isinstance(other, NodeAStar):
            return False
        return (self.distance + self.heuristic_value) > (other.distance + other.heuristic_value)
    
# -------------------------------------------- CLASE SOLVER -------------------------------------------------------#                  
class RubikSolver(RubikCube):
    def __init__(self):
        super().__init__()

    # Herencia de la funcion __move_cube
    def apply_move(self,move):
        if move == 0:
            super().move_xup(1)
        elif move == 1:
            super().move_xbottom(1)
        elif move == 2:
            super().move_xmiddle(1)
        elif move == 3:
            super().move_yleft(1)
        elif move == 4:
            super().move_ymiddle(1)
        elif move == 5:
            super().move_yright(1)
        elif move == 6:
            super().move_zback(1)
        elif move == 7:
            super().move_zfront(1)
        elif move == 8:
            super().move_zmiddle(1)
        elif move == 9:
            super().move_xup(3)
        elif move == 10:
            super().move_xbottom(3)
        elif move == 11:
            super().move_xmiddle(3)
        elif move == 12:
            super().move_yleft(3)
        elif move == 13:
            super().move_ymiddle(3)
        elif move == 14:
            super().move_yright(3)
        elif move == 15:
            super().move_zback(3)
        elif move == 16:
            super().move_zfront(3)
        elif move == 17:
            super().move_zmiddle(3)

    # 54 arreglo de 6 bits para optimizar
    def encode(self, cube):
        # Representación binaria de cada color: blanco-rojo-azul-verde-naranja-amarillo
        # 100000 - 010000 - 001000 - 000100 - 000010 - 000001
            encoded_state = []
            aux = [] 
            count = 0
            for face in cube:
                for row in face:
                    for elem in row:
                        if elem == 0:  # Blanco
                            encoded_state.append([1, 0, 0, 0, 0, 0])
                        elif elem == 1:  # Rojo
                            encoded_state.append([0, 1, 0, 0, 0, 0])
                        elif elem == 2:  # Azul
                            encoded_state.append([0, 0, 1, 0, 0, 0])
                        elif elem == 3:  # Verde
                            encoded_state.append([0, 0, 0, 1, 0, 0])
                        elif elem == 4:  # Naranja
                            encoded_state.append([0, 0, 0, 0, 1, 0])
                        else:  # Amarillo
                            encoded_state.append([0, 0, 0, 0, 0, 1])
                        '''
                        count += 1
                        if count % 9 == 0:  
                            encoded_state.append(aux)
                            aux = [] 
                        '''
            return encoded_state
    

    #------------------------------------------------------------ BFS (Breadth-First-Search)--------------------------------------------------------# 
    def breadth_first_search(self, initial_state, solved_state):
            visited = set()
            queue = [(initial_state, [])]

            while queue:
                state, path = queue.pop(0)
                if self.encode(state.cube) == self.encode(solved_state.cube):
                    return len(path), path

                encoded_state = self.encode(state.cube)
                visited.add(tuple(map(tuple, encoded_state)))

                for move in range(18):
                    new_state = copy.deepcopy(state)
                    new_state.apply_move(move)

                    encoded_new_state = self.encode(new_state.cube)
                    if tuple(map(tuple, encoded_new_state)) not in visited:
                        queue.append((new_state, path + [move]))

            return None
    
    #------------------------------------------------------------  BFS (Best-First-Search) --------------------------------------------------------# 
    def best_first_search(self, initial_state, solved_state, heuristic):
        visited = set()
        queue = deque()
        source = Node(initial_state, path=[])
        target = Node(solved_state)
        source.calculate_heuristic(target, heuristic)
        queue.append(source)
        while queue:
            current_node = queue.popleft()
            encoded_state = tuple(map(tuple, self.encode(current_node.state.cube)))

            if encoded_state == tuple(map(tuple, self.encode(solved_state.cube))):
                return len(current_node.path), current_node.path
            visited.add(encoded_state)

            for move in range(18):  # Hay 18 movimientos posibles en un cubo de Rubik
                new_state = copy.deepcopy(current_node.state)
                new_state.apply_move(move)
                encoded_new_state = tuple(map(tuple, self.encode(new_state.cube)))
                if encoded_new_state not in visited:
                    new_node = Node(new_state, path=current_node.path + [move])
                    new_node.calculate_heuristic(target, heuristic)
                    queue.append(new_node)

        return None


    #------------------------------------------------------------  A* --------------------------------------------------------# 
    def a_star(self, initial_state, solved_state, heuristic):
        pass




    def new_method(self, cube):
        pass




# ----------------------------CASO RPUEBA----------------------------------------#
    
cubo = RubikSolver()
'''
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

cubo = RubikSolver()
cubo_resuelto = RubikSolver()  # Se inicializa un nuevo cubo resuelto
shuffled_state = cubo.shuffle(1)  # Se obtiene el estado del cubo revuelto después del shuffle
print("\nCubo a resolver revuelto al azar:")
cubo.print_cube()'''

'''print("----------breadhtfs-----------------")
print("Cantidad de movimientos y lista de movimientos para resolver el cubo:")
movimientos_necesarios, movimientos = cubo.breadth_first_search(shuffled_state, cubo_resuelto)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)
cubo.print_cube()'''
print("----------bestfs-----------------")
solved_state = RubikSolver()  
initial_state = cubo.shuffle(2)  # Se obtiene el estado del cubo revuelto después del shuffle
cubo.print_cube()
movimientos_necesarios, movimientos = cubo.best_first_search(initial_state, solved_state, Heuristica.bfs)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)