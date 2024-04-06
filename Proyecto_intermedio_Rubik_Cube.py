import random
from collections import deque
from queue import Queue
import copy
from queue import PriorityQueue
import numpy as np

class RubikCube:
    def __init__(self):
        # 0 - Cara Frente - BLANCO
        # 4 - Cara derecha - ROJO
        # 1 - Cara superior - AZUL
        # 2 - Cara inferior - VERDE
        # 3 - Cara izquierda - NARANJA
        # 5 - Cara trasera - AMARILLO

        self.cube = [0,0,0,0,0,0]

        # 000 - white
        # 001 - blue
        # 010 - green
        # 011 - orange
        # 100 - red
        # 101 - yellow
        
        self.colors = [0,1,2,3,4,5]

        self.rotation = [[(18, 20), (9, 11), (0,2)],
                         [(21, 23), (12, 14), (3, 5)],
                         [(24, 26), (15, 17), (6, 8)]]
        
        #Inicializa el cubo
        for i in range(len(self.cube)):
            self.cube[i] = self.colors[i]
            for _ in range(8):
                self.cube[i] = (self.cube[i]<<3) | self.colors[i]
        
    # "Y" HACIA ARRIBA
    # 0 - 2 - 5 - 3

    # "X" HACIA ARRIBA
    # 0 - 1 - 5 - 4

    # "Z" HACIA ARRIBA
    # 2 - 1 - 3 - 4

    def rotate(self, n):
        for i in range(2, 0):
            for j in range(2, 0):
                mask = (mask << 3) | self.__get_bits(n, self.rotation[i][j][0], self.rotation[i][j][1]) # n = la cara que se quiere rotar

        #antihorario - arriba a abajo, izquierda derecha
        #horario - derecha a izquierda, abajo a arriba
        
        return mask

    def __get_bits(self, n, lb, gb): # El número y el rango 
        if lb > gb or gb < lb:
            return 0
        
        mask = ( (2 ** (gb + 1) - 1) >> lb) << lb 

        return (n & mask) >> lb # Se regresa el número en bits
    

    def __set_bits(self, n, bits, lb, gb, max_bits = 26): # En el número donde le voy a meter los bits de donde a donde y el máximo de bits
        mask1 = self.__get_bits(n, gb +1, max_bits) # Antes de donde van los bits
        mask2 = bits # Los bits que se ponen el lugar donde se ponen
        mask3 = self.__get_bits(n, 0, lb - 1) # Los bits desde el lb hasta 0
        num = ((mask1 << (gb - lb + 1) | mask2) << lb) | mask3 # | (or) mete los bits, se van corriendo los bits para que quepan

        return num
    

    def move_xup(self, n_moves):
        front = self.__get_bits(self.cube[0], 0, 8)
        right = self.__get_bits(self.cube[4], 0, 8)
        back = self.__get_bits(self.cube[5], 0, 8)
        left = self.__get_bits(self.cube[3], 0, 8)

        self.cube[0] = self.__set_bits(self.cube[0], left, 0, 8)
        self.cube[4] = self.__set_bits(self.cube[4], front, 0, 8)
        self.cube[5] = self.__set_bits(self.cube[5], right, 0, 8)
        self.cube[3] = self.__set_bits(self.cube[3], back, 0, 8)
    
    def move_xmiddle(self, n_moves):#C
        front = self.__get_bits(self.cube[0], 18, 26)
        right = self.__get_bits(self.cube[4], 18, 26)
        back = self.__get_bits(self.cube[5], 18, 26)
        left = self.__get_bits(self.cube[3], 18, 26)

        self.cube[0] = self.__set_bits(self.cube[0], left, 18, 26)
        self.cube[4] = self.__set_bits(self.cube[4], front, 18, 26)
        self.cube[5] = self.__set_bits(self.cube[5], right, 18, 26)
        self.cube[3] = self.__set_bits(self.cube[3], back, 18, 26)
        
    def move_xbottom(self, n_moves):#AP
        front = self.__get_bits(self.cube[0], 18, 26)
        right = self.__get_bits(self.cube[4], 18, 26)
        back = self.__get_bits(self.cube[5], 18, 26)
        left = self.__get_bits(self.cube[3], 18, 26)

        self.cube[0] = self.__set_bits(self.cube[0], left, 18, 26)
        self.cube[4] = self.__set_bits(self.cube[4], front, 18, 26)
        self.cube[5] = self.__set_bits(self.cube[5], right, 18, 26)
        self.cube[3] = self.__set_bits(self.cube[3], back, 18, 26)

    # Gira la columna izquierda en sentido horario
    def move_yleft(self, n_moves):
        # se llama al get_gits para cada parámetro del front , top ... y luego al set_bit de los del front al del top como corresponden, se calcula el y_left y se le mandas un delta + 6 para la derecha
        pass

    def move_yright(self, n_moves):
        pass


        # Gira la cara trasera en sentido horario
    def move_zfront(self, n_moves):
        pass

    # Gira la cara trasera en sentido horario
    def move_zback(self, n_moves):
        pass




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
            self.__move_cube(3)
            self.__move_cube(14)
        elif move == 5:
            self.move_yright(1)
        elif move == 6:
            self.move_zback(1)
        elif move == 7:
            self.move_zfront(1)
        elif move == 8:
            self.__move_cube(16)
            self.__move_cube(6)
        elif move == 9:
            self.move_xup(3)
        elif move == 10:
            self.move_xbottom(3)
        elif move == 11:
            self.move_xmiddle(3)
        elif move == 12:
            self.move_yleft(3)
        elif move == 13:
            self.__move_cube(12)
            self.__move_cube(5)
        elif move == 14:
            self.move_yright(3)
        elif move == 15:
            self.move_zback(3)
        elif move == 16:
            self.move_zfront(3)
        elif move == 17:
            self.__move_cube(7)
            self.__move_cube(15)
            

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
    def heuristic0(node_a, node_b):
        return 0
    
    @staticmethod
    def esquinas(node_a, node_b):
        cube_uno = node_a.state.cube
        cube_dos = node_b.state.cube
        misplaced_edges = 0

        # Va checando las esquinas que no están bien
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for face in range(6):
            for edge in edges:
                row, col = edge
                if cube_uno[face][row][col] != cube_dos[face][row][col]:
                    misplaced_edges += 1

        return misplaced_edges
            
    @staticmethod
    def color_centro(node_a, node_b):
        state = node_a.state.cube
        target_state = node_b.state.cube
        distancia = 0

        for face in range(6):
            color_centro = state[face][1][1]  
            target_color_centro = target_state[face][1][1] 
            if color_centro != target_color_centro:
                distancia += 1

        return distancia

    
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
            self.apply_move(3)
            self.apply_move(14)
        elif move == 5:
            super().move_yright(1)
        elif move == 6:
            super().move_zback(1)
        elif move == 7:
            super().move_zfront(1)
        elif move == 8:
            self.apply_move(16)
            self.apply_move(6)
        elif move == 9:
            super().move_xup(3)
        elif move == 10:
            super().move_xbottom(3)
        elif move == 11:
            super().move_xmiddle(3)
        elif move == 12:
            super().move_yleft(3)
        elif move == 13:
            self.apply_move(12)
            self.apply_move(5)
        elif move == 14:
            super().move_yright(3)
        elif move == 15:
            super().move_zback(3)
        elif move == 16:
            super().move_zfront(3)
        elif move == 17:
            self.apply_move(7)
            self.apply_move(15)

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
        visited = set()
        pq = PriorityQueue()
        source = NodeAStar(initial_state, path=[])
        target = NodeAStar(solved_state)
        source.calculate_heuristic(target, heuristic)
        pq.put((source.heuristic_value, source))

        while not pq.empty():
            _, current_node = pq.get()
            encoded_state = tuple(map(tuple, self.encode(current_node.state.cube)))

            if encoded_state == tuple(map(tuple, self.encode(solved_state.cube))):
                return len(current_node.path), current_node.path

            visited.add(encoded_state)

            for move in range(18):  # Hay 18 movimientos posibles en un cubo de Rubik
                new_state = copy.deepcopy(current_node.state)
                new_state.apply_move(move)
                encoded_new_state = tuple(map(tuple, self.encode(new_state.cube)))
                if encoded_new_state not in visited:
                    new_node = NodeAStar(new_state, path=current_node.path + [move])
                    new_node.distance = current_node.distance + 1
                    new_node.calculate_heuristic(target, heuristic)
                    pq.put((new_node.heuristic_value + new_node.distance, new_node))

        return None
    
    def iterative_deepening_depth_first_search(self, max_depth, solved_state, heuristic):
            for depth in range(1, max_depth + 1):
                result = self.depth_limited_search(depth, solved_state, [], heuristic)
                if result is not None:
                    print("cantidad de movimientos:", len(result))
                    return result
            return None

    def depth_limited_search(self, depth_limit, solved_state, path, heuristic):
        if depth_limit == 0:
            return None
        if self.cube == solved_state.cube:  
            return path
        moves = list(range(18))
        moves.sort(key=lambda move: heuristic(Node(self), Node(solved_state)))
        for move in range(18):
            new_cube = copy.deepcopy(self)  
            new_cube.apply_move(move)
            result = new_cube.depth_limited_search(depth_limit - 1, solved_state, path + [move], heuristic)
            if result is not None:
                return result
        return None




cubo = RubikCube()

cubo.shuffle(None, [7])
#cubo.display_cube()
cubo.print_cube()


# ----------------------------CASO RPUEBA----------------------------------------#
    
print("\n*******Movimientos que puedes hacer*******\n")
print("0) Mover fila superior a la derecha")
print("1) Mover fila inferior a la derecha")
print("2) Mover fila intermedia a la izquierda")

print("3) Mover columna izquierda hacia arriba")
print("4) Mover columna intermedia hacia abajo")
print("5) Mover columna derecha hacia arriba")

print("6) Mover cara trasera en sentido a las manecillas del reloj")
print("7) Mover cara frontal en sentido de las manecillas del reloj")
print("8) Mover cara intermedia en contrasentido de las manecillas del reloj")

print("9) Mover fila superior a la izquierda")
print("10) Mover fila inferior a la izquierda")
print("11) Mover fila intermedia a la derecha")

print("12) Mover columna izquierda hacia abajo")
print("13) Mover columna intermedia hacia arriba")
print("14) Mover columna derecha hacia abajo")

print("15) Mover cara trasera en contrasentido a las manecillas del reloj")
print("16) Mover cara frontal en contrasentido de las manecillas del reloj")
print("17) Mover cara intermedia en sentido de las manecillas del reloj")
print("Cubo original:")
'''
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
'''
cubo = RubikSolver()
'''
cubo_resuelto = RubikSolver()  # Se inicializa un nuevo cubo resuelto
shuffled_state = cubo.shuffle(None, [4])  # Se obtiene el estado del cubo revuelto después del shuffle
print("\nCubo a resolver revuelto al azar:")
cubo.print_cube()

print("----------breadhtfs-----------------")
print("Cantidad de movimientos y lista de movimientos para resolver el cubo:")
movimientos_necesarios, movimientos = cubo.breadth_first_search(shuffled_state, cubo_resuelto)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)
cubo.print_cube()
'''
'''print("----------bestfs-----------------")
solved_state = RubikSolver()  

initial_state = cubo.shuffle(None, [8, 10])  # Se obtiene el estado del cubo revuelto después del shuffle
cubo.print_cube()
movimientos_necesarios, movimientos = cubo.best_first_search(initial_state, solved_state, Heuristica.bfs)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)'''

'''print("----------A*-----------------")
solved_state = RubikSolver()  
movimientos_manual = [0, 8, 10, 11]
initial_state = cubo.shuffle(None, movimientos_manual)  # Se obtiene el estado del cubo revuelto después del shuffle
cubo.print_cube()
movimientos_necesarios, movimientos = cubo.a_star(initial_state, solved_state, Heuristica.color_centro)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)'''


'''print("----------Iterative-deepening-----------------")
rubik = RubikSolver()
goal_state = RubikSolver()  # Estado objetivo resuelto

# Mezclar el cubo de Rubik

rubik.shuffle(None, [7, 4, 9, 14])
rubik.print_cube()
# Resolver el cubo de Rubik
solution = rubik.iterative_deepening_depth_first_search(8, goal_state, Heuristica.heuristic0)
print("Solución encontrada:",solution)'''