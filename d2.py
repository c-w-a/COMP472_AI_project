# DELIVERABLE 2
from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import math
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

class HeuristicType(Enum):
    E0 = 0
    E1 = 1

##############################################################################################################

@dataclass()
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    Max_health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9
    
    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass()
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    def iter_diagonal(self) -> Iterable[Coord]:
        """ "Iterates over diagonal Coords."""
        yield Coord(self.row - 1, self.col - 1)
        yield Coord(self.row - 1, self.col + 1)
        yield Coord(self.row + 1, self.col - 1)
        yield Coord(self.row + 1, self.col + 1)
    def iter_adjacent_and_diagonal(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)
        yield Coord(self.row+1,self.col+1)
        yield Coord(self.row+1,self.col-1)
        yield Coord(self.row-1,self.col+1)
        yield Coord(self.row-1,self.col-1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = False
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    search_depth : int = 4
    heuristic : HeuristicType = HeuristicType.E0

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    ai_attacker : bool = True
    ai_defender : bool = True
    

    def __post_init__(self):
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self.ai_attacker = False
                else:
                    self.ai_defender = False

    def check_dead(self):
        all_coords = CoordPair(Coord(0, 0), Coord(4, 4)).iter_rectangle()
        for coord in all_coords:
            if self.get(coord) is not None:
                unit = self.get(coord)
                if not unit.is_alive():
                    self.remove_dead(coord)

    def mod_health(self, coord : Coord, health_delta : int):
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        "Validate a move expressed as a CoordPair."
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):        #Checks if coord at source is valid and if coord at destination is valid
            return False
       
       
        unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)
        adjacentCoords = Coord.iter_adjacent(coords.src)

        if unit is None or unit.player != self.next_player:
            return False

        if coords.dst not in adjacentCoords and coords.dst != coords.src:
            return False

        if (unit.type not in [UnitType.Virus, UnitType.Tech] and dst_unit is None):  
            for adjacentCoord in adjacentCoords:
                adjacentUnit = self.get(adjacentCoord)
                if adjacentUnit is not None and adjacentUnit.player != unit.player:
                    return False

            if (unit.player == Player.Attacker):  
                if coords.dst.col > coords.src.col or coords.dst.row > coords.src.row:
                    return False
            else: 
                if coords.dst.col < coords.src.col or coords.dst.row < coords.src.row:
                    
                    return False

        if (dst_unit is not None and dst_unit.player == unit.player and not dst_unit == unit): 
            if unit is UnitType.Tech and dst_unit is UnitType.Virus:
                return False

            if (dst_unit.health >= 9):  
                return False

        return True


    def self_destruct(self, coords: CoordPair, source_unit: Unit):
        """Method to self-destruct, damages all surrounding units within range of 1"""
        self.mod_health(coords.src, -source_unit.health)
        for adjacentCoords in coords.src.iter_range(1):
            adjacent_unit = self.get(adjacentCoords)
            if adjacent_unit:
                self.mod_health(adjacentCoords, -2)


    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""
        if self.is_valid_move(coords):
            src_unit = self.get(coords.src)
            dst_unit = self.get(coords.dst)

            if coords.dst == coords.src:
                self.self_destruct(coords, src_unit)
                return True, "Self Destructed"
            if not self.is_empty(coords.dst):
                dst_unit = self.get(coords.dst)
            elif dst_unit and dst_unit.player != src_unit.player:
                t_dmg = src_unit.damage_amount(dst_unit)
                s_dmg = dst_unit.damage_amount(src_unit)
                self.mod_health(coords.src, -s_dmg)
                self.mod_health(coords.dst, -t_dmg)

            elif dst_unit and dst_unit.player == src_unit.player:
                repair = src_unit.repair_amount(dst_unit)
                self.mod_health(coords.dst, repair)
                damage_amount = dst_unit.damage_amount(src_unit)
                self.mod_health(coords.src, -damage_amount)
                return True, f"Attacked unit. New health: {dst_unit.health}"
            else:
                self.set(coords.dst,self.get(coords.src))
                self.set(coords.src,None)
                return True, ""
        return False, "invalid move"
    

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1
        self.check_dead()

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output
    
    def board_config_to_string(self) -> str:
        dim = self.options.dim
        coord = Coord()
        output = ""
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self) -> str:
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    self.next_turn()
                    return result
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        start_time = datetime.now()
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end="")
                print(result)
                self.next_turn()
        elapse_time = (datetime.now()-start_time).total_seconds()
        return mv, elapse_time

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self.ai_attacker:
            if self.ai_defender:
                return None
            else:
                return Player.Attacker    
        return Player.Defender


    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

            

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)
        
    def get_unit_count(self, player: Player, unit_type: UnitType) -> int:
        """Returns the count of a specific unit type for a given player."""
        unit_counter = 0
        for _, unit in self.player_units(player):
            if unit.type == unit_type:
                unit_counter += 1
        return unit_counter
        

    def e0(self) -> int:
        attackerScore = 0
        defenderScore = 0
        score = 0
        if self.next_player == Player.Attacker:
            attackerScore = ((3 * self.get_unit_count(Player.Attacker, UnitType.Virus)) + (3 * self.get_unit_count(Player.Attacker, UnitType.Tech)) + (3 * self.get_unit_count(Player.Attacker, UnitType.Firewall)) + (3 * self.get_unit_count(Player.Attacker, UnitType.Program)) + (9999 * self.get_unit_count(Player.Attacker, UnitType.AI)))
            defenderScore = ((3 * self.get_unit_count(Player.Defender, UnitType.Virus)) + (3 * self.get_unit_count(Player.Defender, UnitType.Tech)) + (3 * self.get_unit_count(Player.Defender, UnitType.Firewall)) + (3 * self.get_unit_count(Player.Defender, UnitType.Program)) + (9999 * self.get_unit_count(Player.Defender, UnitType.AI)))
            score = attackerScore - defenderScore
        return score
    

    def e1(self) -> int:
        # Define attack values for each unit
        ATTACK_POINTS = {
            UnitType.AI: 2,
            UnitType.Tech: 3,
            UnitType.Virus: 9,
            UnitType.Program: 4,
            UnitType.Firewall: 2
        }

        # Determine which units belong to the attacker and which belong to the defender
        if self.next_player == Player.Attacker:
            attackerUnits = self.player_units(self.next_player)
            defenderUnits = self.player_units(self.next_player.next())
        else:
            defenderUnits = self.player_units(self.next_player)
            attackerUnits = self.player_units(self.next_player.next())

        # Calculate total attack power for attacker
        attackerPower = sum([ATTACK_POINTS[unit.type] for coord, unit in attackerUnits])

        # Calculate total attack power for defender
        defenderPower = sum([ATTACK_POINTS[unit.type] for coord, unit in defenderUnits])

        # Return the unit advantage
        return attackerPower - defenderPower
    
    def e2(self, Coords: CoordPair) -> int:
        # Define kamikaze values for unit types (you can adjust these values)
        kamikaze_values = {
            UnitType.Virus: 10,
            UnitType.Tech: 2,
            UnitType.Program: 1,
            UnitType.Firewall: 1,
        }

        ai_unit = Coords.src
        kamikaze_value = kamikaze_values.get(ai_unit.type, 0)

        # Calculate potential damage to adversarial units
        damage_to_adversarial = 0
        for adjacent_coords in ai_unit.iter_range(1):
            adjacent_unit = ai_unit(adjacent_coords)
            if adjacent_unit and adjacent_unit.player != ai_unit.player:
                damage_to_adversarial += 2  # Damage inflicted to adversarial units

        # Calculate potential damage to friendly units
        damage_to_friendly = 0
        for adjacent_coords in ai_unit.iter_range(1):
            adjacent_unit = self.get(adjacent_coords)
            if adjacent_unit and adjacent_unit.player == ai_unit.player:
                damage_to_friendly += 2  # Damage inflicted to friendly units

        # Calculate kamikaze score
        kamikaze_score = kamikaze_value * (damage_to_adversarial - damage_to_friendly)

        return kamikaze_score

    

    
    def e_total(self) -> int:
        """Returns the total heuristic score."""
        return self.e0()
        

    def alpha_beta(self, depth: int, alpha: int, beta: int):
        if depth == 0 or self.is_finished():
            return (self.e0(), None)
        leaf = self.get_leaf(self.next_player)
        if self.next_player == Player.Attacker:  
            max_eval = -(math.inf)
            maxMove = None
            for (game_copy, move) in leaf:
                if maxMove is None:
                    maxMove = move
                eval = game_copy.alpha_beta(depth-1, alpha, beta)[0]
                max_eval = max(eval, max_eval)
                if eval > max_eval:
                    max_eval = eval
                    maxMove = move
                if (self.options.alpha_beta):
                    alpha = max(alpha, max_eval)
                    if beta <= alpha:
                        break 
            return (max_eval, maxMove)
        else:  
            min_eval = math.inf
            minMove = None
            for (game_copy, move) in leaf:
                if minMove is None:
                    minMove = move
                eval = game_copy.alpha_beta(depth-1, alpha, beta)[0]
                if eval <= min_eval:
                    min_eval = eval
                    minMove = move
                if (self.options.alpha_beta):
                    beta = min(beta, min_eval)
                    if beta <= alpha:
                        break  
            return (min_eval, minMove)



    def minimax(self, depth, maximizing_player):
        if depth == 0 or self.is_finished():
            return self.e0()

        if maximizing_player:
            max_eval = float("-inf")
            for (child) in self.get_leaf(Player.Attacker):
                eval = child.minimax(depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for (child) in self.get_leaf(Player.Defender):
                eval = child.minimax(depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
        

    def get_leaf(self, player: Player) -> Iterable[Tuple[Game, CoordPair] | None]:
        leaf = []
        for (coord, unit) in self.player_units(player):
            adjacentCoords = Coord.iter_adjacent(coord)
            for adjacentCoord in adjacentCoords:
                game_copy = self.clone()
                move = CoordPair(coord, adjacentCoord)
                if game_copy.perform_move(move)[0]:
                    game_copy.next_turn()
                    leaf.append((game_copy, move))
            selfdestruct_move = CoordPair(coord, coord)
            if game_copy.perform_move(selfdestruct_move)[0]:
                game_copy.next_turn()
                leaf.append((game_copy, selfdestruct_move))
        return leaf

    def suggest_move(self) -> CoordPair | None:
        start_time = datetime.now()
        (score, move) = self.alpha_beta(self.options.max_depth, - (math.inf), math.inf)  # Removed avg_depth
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        if (elapsed_seconds > self.options.max_time):
            print(
                f"{self.next_player.name} has taken too much time, {self.next_player.next().name} wins!")
        self.stats.total_seconds += elapsed_seconds
        print(f"Suggested move: {move} with score of {score} ")
        print(f"Heuristic score: {self.e0()}")
        print(f"Evals per depth: ", end="")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end="")
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")

        return move
   

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        pass
                else:
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def str_to_bool(str):
    if str.lower() in ('true'):
        return True
    elif str.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--alpha_beta', type=str_to_bool, help='True to turn on alpha-beta pruning')
    parser.add_argument('--max_turns', type=int, help='maximum turns')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--heuristic', type=int, help='choose which heuristic to use (0,1, or 2)') 
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp


    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.broker is not None:
        options.broker = args.broker
    if args.heuristic is not None:
        options.heuristic = args.heuristic

    game = Game(options=options)

    # create output file and output all the game parameter stuff
    filename = 'gameTrace-' + str(game.options.alpha_beta) + '-' + str(int(game.options.max_time)) + '-' + str(game.options.max_turns) + '.txt'
    out_file = open(filename, 'w')
    out_file.write("\n --- GAME PARAMETERS --- \n\n")
    out_file.write("t = " + str(game.options.max_time) + "s\n")
    out_file.write("max # of turns: " + str(game.options.max_turns) + "\n")
    out_file.write("play mode: " + str(game.options.game_type)[9:] + "\n")
    if game.options.game_type != GameType.AttackerVsDefender:
        out_file.write("alpha-beta: " + str(game.options.alpha_beta) + "\n")
        out_file.write("heuristic: " + str(game.options.heuristic)[14:] + "\n")
    out_file.write("\n\n --- INITIAL BOARD CONFIG ---\n")
    out_file.write(game.board_config_to_string())
    out_file.write('\n\n --- TURNS ---\n\n')

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            out_file.write('\n --- WINNER --- \n\n')
            out_file.write(winner.name + ' wins in ' + str(game.turns_played))
            if game.turns_played == 1:
                out_file.write(' turn!\n')
            else:
                out_file.write(' turns!\n')
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            result = game.human_turn()
            if game.next_player == Player.Attacker:
                player = 'Defender'
            else:   
                player = 'Attacker'
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                out_file.close()
                exit(1)
        
        # output info for each turn: TO-DO MAKE IT PRINT THE RIGHT INFO TO CONSOLE TOO
        out_file.write('turn #' + str(game.turns_played) + '\n')
        out_file.write('player: ' + player + '\n')
        out_file.write('action: ' + result)
        out_file.write(game.board_config_to_string() + '\n')

##############################################################################################################

if __name__ == '__main__':
    main()
