import random
from queue import Queue
import copy
from queue import PriorityQueue
import time
class RubikCube:
    def __init__(self):
        # 0 - Cara Frente - BLANCO
        # 4 - Cara derecha - ROJO
        # 1 - Cara superior - AZUL
        # 2 - Cara inferior - VERDE
        # 3 - Cara izquierda - NARANJA
        # 5 - Cara trasera - AMARILLO

        self.cube = [0, 0, 0, 0, 0, 0]
        # 000-blanco/ 001-azul / 010-verde / 011-naranja / 100-rojo / 101-amarillo

        self.colors = [0, 1, 2, 3, 4, 5]

        self.rotation = [[(18, 20), (9, 11), (0, 2)],
                         [(21, 23), (12, 14), (3, 5)],
                         [(24, 26), (15, 17), (6, 8)]]

        # Inicializa el cubo
        for i in range(len(self.cube)):
            self.cube[i] = self.colors[i]
            for _ in range(8):
                self.cube[i] = (self.cube[i] << 3) | self.colors[i]

    # "Y" HACIA ARRIBA ||  "X" HACIA ARRIBA  || "Z" HACIA ARRIBA
    # 0 - 2 - 5 - 3    ||   0 - 1 - 5 - 4    ||   2 - 1 - 3 - 4

    def rotate_antihorario(self, n):
        mask = 0
        for i in range(3):
            for j in range(3):
                mask = (mask << 3) | self.__get_bits(n, self.rotation[i][j][0], self.rotation[i][j][1])
        # antihorario - arriba a abajo, izquierda derecha

        return mask
    
    def rotate_horario(self, n):
        mask = 0
        for i in range(2, -1, -1):
            for j in range(2, -1, -1):
                mask = (mask << 3) | self.__get_bits(n, self.rotation[i][j][0], self.rotation[i][j][1])
        # horario - derecha a izquierda, abajo a arriba
        return mask

    def __get_bits(self, n, lb, gb):
        if lb > gb or gb < lb:
            return 0

        mask = ((2 ** (gb + 1) - 1) >> lb) << lb

        return (n & mask) >> lb  # Se regresa el número en bits

    def __set_bits(self, n, bits, lb, gb, max_bits=26):
        mask1 = self.__get_bits(n, gb + 1, max_bits)  # Antes de donde van los bits
        mask2 = bits  # Los bits que se ponen el lugar donde se ponen
        mask3 = self.__get_bits(n, 0, lb - 1)  # Los bits desde el lb hasta 0
        num = ((mask1 << (gb - lb + 1) | mask2) << lb) | mask3  # | (or) mete los bits, se van corriendo los bits para que quepan

        return num

    # ----------------------------Movimientos en x--------------------------------------
    def move_xup(self, n_moves):
        for _ in range(n_moves):
            front = self.__get_bits(self.cube[0], 0, 8)
            right = self.__get_bits(self.cube[4], 0, 8)
            back = self.__get_bits(self.cube[5], 0, 8)
            left = self.__get_bits(self.cube[3], 0, 8)

            self.cube[0] = self.__set_bits(self.cube[0], left, 0, 8)
            self.cube[4] = self.__set_bits(self.cube[4], front, 0, 8)
            self.cube[5] = self.__set_bits(self.cube[5], right, 0, 8)
            self.cube[3] = self.__set_bits(self.cube[3], back, 0, 8)

            self.cube[1]=self.rotate_antihorario(self.cube[1])

    def move_xbottom(self, n_moves):
        for _ in range(n_moves):
            front = self.__get_bits(self.cube[0], 18, 26)
            right = self.__get_bits(self.cube[4], 18, 26)
            back = self.__get_bits(self.cube[5], 18, 26)
            left = self.__get_bits(self.cube[3], 18, 26)

            self.cube[0] = self.__set_bits(self.cube[0], left, 18, 26)
            self.cube[4] = self.__set_bits(self.cube[4], front, 18, 26)
            self.cube[5] = self.__set_bits(self.cube[5], right, 18, 26)
            self.cube[3] = self.__set_bits(self.cube[3], back, 18, 26)

            self.cube[2]=self.rotate_horario(self.cube[2])

    # ------------------------------------Movimientos en y-------------------------------
    def move_yleft(self, n):
        for _ in range(n):
            front = []
            up = []
            back = []
            bottom = []

            # Obtener los bits de cada cara
            front.append(self.__get_bits(self.cube[0], 0, 2))
            front.append(self.__get_bits(self.cube[0], 9, 11))
            front.append(self.__get_bits(self.cube[0], 18, 20))

            up.append(self.__get_bits(self.cube[1], 0, 2))
            up.append(self.__get_bits(self.cube[1], 9, 11))
            up.append(self.__get_bits(self.cube[1], 18, 20))

            back.append(self.__get_bits(self.cube[5], 6, 8))
            back.append(self.__get_bits(self.cube[5], 15, 17))
            back.append(self.__get_bits(self.cube[5], 24, 26))

            bottom.append(self.__get_bits(self.cube[2], 0, 2))
            bottom.append(self.__get_bits(self.cube[2], 9, 11))
            bottom.append(self.__get_bits(self.cube[2], 18, 20))

            # Establecer los bits en las nuevas caras
            self.cube[0] = self.__set_bits(self.cube[0], bottom[0], 0, 2)
            self.cube[0] = self.__set_bits(self.cube[0], bottom[1], 9, 11)
            self.cube[0] = self.__set_bits(self.cube[0], bottom[2], 18, 20)

            self.cube[1] = self.__set_bits(self.cube[1], front[0], 0, 2)
            self.cube[1] = self.__set_bits(self.cube[1], front[1], 9, 11)
            self.cube[1] = self.__set_bits(self.cube[1], front[2], 18, 20)

            self.cube[5] = self.__set_bits(self.cube[5], up[0], 24, 26)
            self.cube[5] = self.__set_bits(self.cube[5], up[1], 15, 17)
            self.cube[5] = self.__set_bits(self.cube[5], up[2], 6, 8)

            self.cube[2] = self.__set_bits(self.cube[2], back[0], 18, 20)
            self.cube[2] = self.__set_bits(self.cube[2], back[1], 9, 11)
            self.cube[2] = self.__set_bits(self.cube[2], back[2], 0, 2)
            # Rotar la cara izquierda
            self.cube[3]=self.rotate_antihorario(self.cube[3])

    def move_yright(self, n):
        for _ in range(n):
            front = []
            up = []
            back = []
            bottom = []

            # Obtener los bits de cada cara
            front.append(self.__get_bits(self.cube[0], 6, 8))
            front.append(self.__get_bits(self.cube[0], 15, 17))
            front.append(self.__get_bits(self.cube[0], 24, 26))

            up.append(self.__get_bits(self.cube[1], 6, 8))
            up.append(self.__get_bits(self.cube[1], 15, 17))
            up.append(self.__get_bits(self.cube[1], 24, 26))

            back.append(self.__get_bits(self.cube[5], 0, 2))
            back.append(self.__get_bits(self.cube[5], 9, 11))
            back.append(self.__get_bits(self.cube[5], 18, 20))

            bottom.append(self.__get_bits(self.cube[2], 6, 8))
            bottom.append(self.__get_bits(self.cube[2], 15, 17))
            bottom.append(self.__get_bits(self.cube[2], 24, 26))

            # Establecer los bits en las nuevas caras
            self.cube[0] = self.__set_bits(self.cube[0], bottom[0], 6, 8)
            self.cube[0] = self.__set_bits(self.cube[0], bottom[1], 15, 17)
            self.cube[0] = self.__set_bits(self.cube[0], bottom[2], 24, 26)

            self.cube[1] = self.__set_bits(self.cube[1], front[0], 6, 8)
            self.cube[1] = self.__set_bits(self.cube[1], front[1], 15, 17)
            self.cube[1] = self.__set_bits(self.cube[1], front[2], 24, 26)

            self.cube[5] = self.__set_bits(self.cube[5], up[0], 18, 20)
            self.cube[5] = self.__set_bits(self.cube[5], up[1], 9, 11)
            self.cube[5] = self.__set_bits(self.cube[5], up[2], 0, 2)

            self.cube[2] = self.__set_bits(self.cube[2], back[0], 24, 26)
            self.cube[2] = self.__set_bits(self.cube[2], back[1], 15, 17)
            self.cube[2] = self.__set_bits(self.cube[2], back[2], 6, 8)

            # Rotar la cara derecha
            self.cube[4]=self.rotate_horario(self.cube[4])

    # ----------------------------------Movimientos en z--------------------------------------------
    def __move_z(self, n, linea_vertical, linea_horizontal, shift):
        delta_ver = -18 * shift
        delta_hor = 6 * shift

        for _ in range(n):
            up_bits = []
            right_bits = []
            bottom_bits = []
            left_bits = []

            for i in range(len(linea_vertical)):
                up_bits.append(self.__get_bits(self.cube[1], linea_vertical[i][0], linea_vertical[i][1]))
                right_bits.append(self.__get_bits(self.cube[4], linea_horizontal[i][0], linea_horizontal[i][1]))
                bottom_bits.append(self.__get_bits(self.cube[2], linea_vertical[i][0] + delta_ver, linea_vertical[i][1] + delta_ver))
                left_bits.append(self.__get_bits(self.cube[3], linea_horizontal[i][0] + delta_hor, linea_horizontal[i][1] + delta_hor))

            for i in range(len(linea_horizontal)):
                self.cube[1] = self.__set_bits(self.cube[1], left_bits[-(i + 1)], linea_vertical[i][0], linea_vertical[i][1])
                self.cube[4] = self.__set_bits(self.cube[4], up_bits[i], linea_horizontal[i][0], linea_horizontal[i][1])
                self.cube[2] = self.__set_bits(self.cube[2], right_bits[i], linea_vertical[-(i + 1)][0] + delta_ver, linea_vertical[-(i + 1)][1] + delta_ver)
                self.cube[3] = self.__set_bits(self.cube[3], bottom_bits[i], linea_horizontal[i][0] + delta_hor, linea_horizontal[i][1] + delta_hor)

    def move_zfront(self, n):
        # linea vertical              #linea horizontal
        self.__move_z(n, [(18, 20), (21, 23), (24, 26)], [(0, 2), (9, 11), (18, 20)], 1)
        self.cube[0]=self.rotate_horario(self.cube[0])
        

    def move_zback(self, n):
        self.__move_z(n, [(0, 2), (3, 5), (6, 8)], [(6, 8), (15, 17), (24, 26)], -1)
        self.cube[5]=self.rotate_antihorario(self.cube[5])

    # ---------------------------------------mov----------------------------------------------------
    def __move_cube(self, move):
        if move == 0:  # x arriba
            self.move_xup(1)
        elif move == 1:  # x abajo
            self.move_xbottom(1)
        elif move == 2:  # y izquierda
            self.move_yleft(1)
        elif move == 3:  # y derecha
            self.move_yright(1)
        elif move == 4:  # z frontal
            self.move_zfront(1)
        elif move == 5:  # z trasera
            self.move_zback(1)
        if move == 6:  # x arriba
            self.move_xup(3)
        elif move == 7:  # x abajo
            self.move_xbottom(3)
        elif move == 8:  # y izquierda
            self.move_yleft(3)
        elif move == 9:  # y derecha
            self.move_yright(3)
        elif move == 10:  # z frontal
            self.move_zfront(3)
            #self.cube[0]=self.rotate_horario(self.cube[0])
        elif move == 11:  # z trasera
            self.move_zback(3)
            #self.cube[5]=self.rotate_antihorario(self.cube[5])

    def print_cube(self):
        mask = (2 ** 3) - 1
        faces = []
        print(" Cara Frontal         Cara Superior       Cara Inferior       Cara Izquierda       Cara Derecha       Cara Trasera")
        for i in range(len(self.cube)):
            aux = self.cube[i]
            face = []
            for j in range(3):
                row = []
                for k in range(3):
                    row.append("| " + str(aux & mask) + " |")
                    aux >>= 3
                face.append(row)
            faces.append(face)

        for i in range(3):
            for face in faces:
                print(' '.join(face[i]), end='   ')
            print()
        print()

            
    def shuffle(self, n_moves, make_moves=[]):
        if len(make_moves) > 0:
            for move in make_moves:
                self.__move_cube(move)
                
        else: 
            for i in range(n_moves):
                move = random.randint(0, 11)
                self.__move_cube(move)
        return self

#----------------------------------------------------CLASE HEURISTICA----------------------------------------------------------

class Heuristica:
    @staticmethod
    def bfs(node_a, node_b):
        cubo_uno = node_a.state.cube
        cubo_dos = node_b.state.cube
        distancia = 0

        for face in range(6):
            for i in range(3):
                for j in range(3):
                    valor_cubouno = (cubo_uno[face] >> ((2 - i) * 9 + (2 - j) * 3)) & 7
                    face_target, row_target, col_target = -1, -1, -1
                    for c in range(6):
                        for f in range(3):
                            for col in range(3):
                                if ((cubo_dos[c] >> ((2 - f) * 9 + (2 - col) * 3)) & 7) == valor_cubouno:
                                    face_target, row_target, col_target = c, f, col
                                    break
                        if face_target != -1:
                            break
                    distancia += abs(face - face_target) + abs(i - row_target) + abs(j - col_target)

        return distancia

    
    def esquinas_y_aristas(node_a, node_b):
        cube_uno = node_a.state.cube
        cube_dos = node_b.state.cube
        misplaced_corners = 0
        misplaced_edges = 0

        # Contar las esquinas mal ubicadas
        for face in range(6):
            for corner in [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2), 
                        (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2)]:
                x, y, z = corner
                if ((cube_uno[face] >> ((2 - x) * 9 + (2 - y) * 3 + (2 - z))) & 7) != ((cube_dos[face] >> ((2 - x) * 9 + (2 - y) * 3 + (2 - z))) & 7):
                    misplaced_corners += 1

        # Contar las aristas mal ubicadas
        for face in range(6):
            for edge in [(0, 0, 1), (0, 1, 0), (0, 1, 2), (0, 2, 1),
                        (1, 0, 0), (1, 0, 2), (1, 2, 0), (1, 2, 2),
                        (2, 0, 1), (2, 1, 0), (2, 1, 2), (2, 2, 1)]:
                x, y, z = edge
                if ((cube_uno[face] >> ((2 - x) * 9 + (2 - y) * 3 + (2 - z))) & 7) != ((cube_dos[face] >> ((2 - x) * 9 + (2 - y) * 3 + (2 - z))) & 7):
                    misplaced_edges += 1

        return misplaced_corners + misplaced_edges
    
    @staticmethod
    def cfop(node_a, node_b):
        cube_one = node_a.state.cube
        cube_two = node_b.state.cube
        misplaced_crosses = 0
        
        # Evaluar la capa cruzada en la cara superior
        for i in range(3):
            if ((cube_one[1] >> (9 + i * 3)) & 7) != 1:  # Color azul en la cara superior
                misplaced_crosses += 1
            if ((cube_two[1] >> (9 + i * 3)) & 7) != 1:  # Color azul en la cara superior
                misplaced_crosses += 1
        
        # Evaluar la capa cruzada en la cara inferior
        for i in range(3):
            if ((cube_one[2] >> (9 + i * 3)) & 7) != 1:  # Color verde en la cara inferior
                misplaced_crosses += 1
            if ((cube_two[2] >> (9 + i * 3)) & 7) != 1:  # Color verde en la cara inferior
                misplaced_crosses += 1
        
        return misplaced_crosses
    
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
            super().move_yleft(1)
        elif move == 3:
            super().move_yright(1)
        elif move == 4:
            super().move_zback(1)
        elif move == 5:
            super().move_zfront(1)
        elif move == 6:
            super().move_xup(3)
        elif move == 7:
            super().move_xbottom(3)
        elif move == 8:
            super().move_yleft(3)
        elif move == 9:
            super().move_yright(3)
        elif move == 10:
            super().move_zback(3)
        elif move == 11:
            super().move_zfront(3)

    #------------------------------------------------------------ BFS (Breadth-First-Search)--------------------------------------------------------#

    def breadth_first_search(self, initial_state, solved_state):
        visited = set()
        queue = Queue()
        queue.put((initial_state, []))

        while not queue.empty():
            state, path = queue.get()
            if state.cube == solved_state.cube:
                return len(path), path

            visited.add(tuple(state.cube))

            for move in range(12):
                # Poda: no aplicar el mismo movimiento dos veces seguidas
                if path and path[-1] == move:
                    continue
                # Poda: no aplicar un movimiento y luego su inverso
                if path and path[-1] == (move + 6) % 12:
                    continue

                new_state = copy.deepcopy(state)
                new_state.apply_move(move)

                if tuple(new_state.cube) not in visited:
                    queue.put((new_state, path + [move]))

        return None

    
    #------------------------------------------------------------  BFS (Best-First-Search) --------------------------------------------------------# 
    
    def best_first_search(self, initial_state, solved_state, heuristic):
        visited = set()
        queue = PriorityQueue()
        source = Node(initial_state, path=[])
        source.calculate_heuristic(Node(solved_state), heuristic)
        queue.put(source)

        while not queue.empty():
            current_node = queue.get()
            if current_node.state.cube == solved_state.cube:
                return len(current_node.path), current_node.path

            visited.add(tuple(current_node.state.cube))

            for move in range(12):
                new_state = copy.deepcopy(current_node.state)
                new_state.apply_move(move)
                if tuple(new_state.cube) not in visited:
                    new_node = Node(new_state, path=current_node.path + [move])
                    new_node.calculate_heuristic(Node(solved_state), heuristic)
                    queue.put(new_node)


        return None

    #-------------------------------------------------------------------  A* --------------------------------------------------------# 
    def a_star(self, initial_state, solved_state, heuristic):
        visited = set()
        pq = PriorityQueue()
        source = NodeAStar(initial_state, path=[])
        target = NodeAStar(solved_state)
        source.calculate_heuristic(target, heuristic)
        pq.put((source.heuristic_value, source))

        while not pq.empty():
            _, current_node = pq.get()
            if current_node.state.cube == solved_state.cube:
                return len(current_node.path), current_node.path

            visited.add(tuple(current_node.state.cube))

            for move in range(12):
                new_state = copy.deepcopy(current_node.state)
                new_state.apply_move(move)
                if tuple(new_state.cube) not in visited:
                    new_node = NodeAStar(new_state, path=current_node.path + [move])
                    new_node.distance = current_node.distance + 1
                    new_node.calculate_heuristic(target, heuristic)
                    pq.put((new_node.heuristic_value + new_node.distance, new_node))

        return None
        
    
    #--------------------------------------------------------Nueva con uso de heuristica------------------------------------------------------------------
    
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
        
        moves = list(range(12))
        moves.sort(key=lambda move: heuristic(Node(self), Node(solved_state)))
        for move in range(12):
            new_cube = copy.deepcopy(self)  
            new_cube.apply_move(move)
            result = new_cube.depth_limited_search(depth_limit - 1, solved_state, path + [move], heuristic)
            if result is not None:
                return result
        return None
    

# ----------------------------CASOs PRUEBA----------------------------------------#
'''    
print("\n*******Movimientos que puedes hacer*******\n")
print("0) Mover fila superior a la derecha")
print("1) Mover fila inferior a la derecha")

print("2) Mover columna izquierda hacia arriba")
print("3) Mover columna derecha hacia arriba")

print("4) Mover cara frontal en sentido a las manecillas del reloj")
print("5) Mover cara trasera en sentido de las manecillas del reloj")

print("6) Mover fila superior a la izquierda")
print("7) Mover fila inferior a la izquierda")

print("8) Mover columna izquierda hacia abajo")
print("9) Mover columna derecha hacia abajo")

print("10) Mover cara frontal en contrasentido a las manecillas del reloj")
print("11) Mover cara trasera en contrasentido de las manecillas del reloj")

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
'''

'''print("----------breadhtfs-----------------")
cubo_resuelto = RubikSolver()  # Se inicializa un nuevo cubo resuelto
shuffled_state = cubo.shuffle(3)  # Se obtiene el estado del cubo revuelto después del shuffle
print("\nCubo a resolver revuelto:")
cubo.print_cube()
print("Cantidad de movimientos y lista de movimientos para resolver el cubo:")
movimientos_necesarios, movimientos = cubo.breadth_first_search(shuffled_state, cubo_resuelto)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
print("Lista de movimientos:", movimientos)'''


print("----------bestfs-----------------")
cubo = RubikSolver()
solved_state = RubikSolver()  
initial_state = cubo.shuffle(None, [4,3])  # Se obtiene el estado del cubo revuelto después del shuffle
cubo.print_cube()
'''
inicio = time.time()
movimientos_necesarios, movimientos = cubo.best_first_search(initial_state, solved_state, Heuristica.esquinas_y_aristas)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
fin = time.time()
print("Lista de movimientos:", movimientos)
print("Tiempo transcurrido: ", fin - inicio,"segundos")


'''
'''
print("----------A*-----------------")
rubik = RubikSolver()  
solved_state = RubikSolver()  

initial_state = rubik.shuffle(4)  # Se obtiene el estado del cubo revuelto después del shuffle
rubik.print_cube()
inicio = time.time()
movimientos_necesarios, movimientos = rubik.a_star(initial_state, solved_state, Heuristica.esquinas_y_aristas)
print("Cantidad de movimientos necesarios:", movimientos_necesarios)
fin = time.time()
print("Lista de movimientos:", movimientos)
print("Tiempo transcurrido: ", fin - inicio,"segundos")


#print("----------Iterative-deepening-----------------")
rubik = RubikSolver()
goal_state = RubikSolver()  # Estado objetivo resuelto
# Mezclar el cubo de Rubik
rubik.shuffle(3)
rubik.print_cube()
# Resolver el cubo de Rubik
inicio = time.time()
solution = rubik.iterative_deepening_depth_first_search(8, goal_state, Heuristica.bfs)
print("Solución encontrada:",solution)
fin = time.time()
print("Tiempo transcurrido: ", fin - inicio,"segundos")'''