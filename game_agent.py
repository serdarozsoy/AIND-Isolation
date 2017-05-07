"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    """ utility - game result """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")
    

    """legal moves for players as (row,column)""" 
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    """same moves options for our player and opponent (row,column)"""
    shared_moves = [x for x in player_moves if x in opponent_moves]

    " current locations for players (row,column) "
    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))

    """len values"""
    total_moves = game.height * game.width      # total blank spaces when start
    total_moves_left = len(game.get_blank_spaces())  #remaining blank spaces at game
    game_progress = 1.000000001 -(total_moves_left / total_moves) #game stage
    same_moves = len(shared_moves)

    """center,edge,corner variables"""
    my_center_factor = 0.0
    opp_center_factor = 0.0
    my_edge_factor=0.0
    opp_edge_factor=0.0
    my_corner_factor=0.0
    opp_corner_factor=0.0
    row, column = own_loc
    row_o, column_o = opp_loc 
    opp_moves= float(len(opponent_moves))
    my_moves = float(len(player_moves))


    
    #Heuristic 1

    """
    
    a = 1.0
    b = 2.5
    c = 50.0
    
    if my_moves>opp_moves: 
        return a*my_moves - b*opp_moves + c*same_moves
    else:
        return a*my_moves - b*opp_moves  
    
    """

    #Heuristic 2

    """
    
    a = 1.0
    b = 2.5

    if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
        my_edge_factor = -10.0
    if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
       opp_edge_factor = 10.0
        
    return a*my_moves - b*opp_moves+my_edge_factor+opp_edge_factor
    """

    #Heuristic 3
    """
   
    a = 1.0
    b = 2.5
    if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
        my_corner_factor = -10.0
    if ((row_o == 0 or row_o == (game.height-1)) and (column_o == 0 or column_o == (game.width-1))):
        my_corner_factor = 10.0

    return a*my_moves - b*opp_moves+my_corner_factor+opp_corner_factor
    """
    
    #Heuristic 4
    """
    
    a = 1 
    b = 3 
    c = 3 

    if game_progress >= 0.15: 

        c = 50.0*game_progress 

    if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
        my_edge_factor = -20.0 /game_progress 
    if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
        opp_edge_factor = 20.0 /game_progress 
    
    return a*my_moves - b*opp_moves +c*same_moves+my_edge_factor+opp_edge_factor
    
    """

    
    

    # Heuristic 5

    """
    if game_progress <= 0.1: 
        a=1.0
        b=3.0 
        c= 1.0
        if (row == (game.height/2) or row == (game.height/2+1) or row == (game.height/2-1)) and (column == (game.width/2) or column == (game.width/2+1) or column == (game.width/2-1)):# my center
            my_center_factor = 20.0*(1.0/game_progress**2)
        if (row_o == (game.height/2) or row_o == (game.height/2+1) or row_o == (game.height/2-1)) and (column_o == (game.width/2) or column_o == (game.width/2+1) or column_o == (game.width/2-1)):# opp center
            opp_center_factor = -20.0*(1.0/game_progress**2) 
    elif game_progress <= 0.40: 
        a=1.0
        b=1.0*4.0*(game_progress)*(opp_moves/(my_moves+0.1)) 
        c=80*(game_progress**2)
        if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
            my_edge_factor = -50.0 * (game_progress**2)
        if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
            opp_edge_factor = 50.0 * (game_progress**2) 
    
    else:
        
        a=5.0
        b=5.0 
        c=1.0 
        if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
            my_edge_factor = -300.0 * (game_progress**2)
        if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
            opp_edge_factor = 300.0 * (game_progress**2)
        

    return a*my_moves - b*opp_moves +c*same_moves+my_edge_factor+opp_edge_factor+my_center_factor+opp_center_factor
   """
    
    

    # Heuristic 6

    
    if game_progress <= 0.1:
        a=1.0
        b=1.0
        c= 1.0
        if (row == (game.height/2) or row == (game.height/2+1) or row == (game.height/2-1)) and (column == (game.width/2) or column == (game.width/2+1) or column == (game.width/2-1)):
            my_center_factor = 20.0*(1.0/game_progress)
            opp_center_factor = -20.0*(1.0/game_progress**2) 
    elif game_progress <= 0.25:
        a=1.0
        b=10.0*(game_progress)
        c=800.0*(game_progress**2)
        if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
            my_edge_factor = -500.0 * (game_progress**2)
        if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
            opp_edge_factor = 500.0 * (game_progress**2)
    

    else:
        a=1.0
        b=2.5  
        c=800.0*(game_progress**2)
        if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
            my_edge_factor = -300.0 * (game_progress**2)
        if (row_o == 0 or row_o == (game.height-1) or column_o == 0 or column_o == (game.width-1)):
            opp_edge_factor = 300.0 * (game_progress**2)

    return a*my_moves - b*opp_moves +c*same_moves #+ my_edge_factor+opp_edge_factor+my_center_factor+opp_center_factor
    
    



    #Heuristic 7

    """
    

    my_center_factor = 1.0
    my_edge_factor=1.0
    a=1.0
    k=0.0
    b=3.0 * (opp_moves/(my_moves+0.1)) 
    c=1
    if game_progress >= 0.4:
        k = 20.0  #additive importance for shared moves
    if (row == (game.height/2) or row == (game.height/2+1) or row == (game.height/2-1)) and (column == (game.width/2) or column == (game.width/2+1) or column == (game.width/2-1)):
        my_center_factor = 2.0
    if (row == 0 or row == (game.height-1) or column == 0 or column == (game.width-1)):
        my_edge_factor = -2.0
    return a*my_moves - b*opp_moves +(c*my_center_factor*my_edge_factor+k)*same_moves 
    """



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)


        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]  # select random between available moves
        best_score = float("-inf")     # initial best score 





        try:

            
            """ search preference minimax or alphabeta """
            search_prf = self.minimax if self.method == 'minimax' else self.alphabeta 

            """     if iterative is True, we start with depth=1 and increment depth with each examination
                otherwise use predefined search depth in __init__
            """

            if self.iterative:
                curr_depth = 1 
                while True:
                    score, move = search_prf(game, curr_depth)
                    if score > best_score:
                        best_move = move
                    curr_depth += 1
            else:
                score, move = search_prf(game, self.search_depth)
                if score > best_score:
                    best_move = move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        
        legal_moves = game.get_legal_moves()

        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self), (-1, -1)

        nodes = []
        min_max_fn = max if maximizing_player else min

        """
        in the turn of maximizing player, it will be maximize, otherwise it will be minimize.
        For each move in list of legal moves, create possible game snapshot and apply minimax method. 
        create an array with resulted score and moves. Return value will take min or max value of score and correspondent move iteratively.
        """
        for move in legal_moves:
            new_game = game.forecast_move(move)
            nodes.append((self.minimax(new_game, depth-1, not maximizing_player)[0], move))
        
        return min_max_fn(nodes, key = lambda m : m[0])     

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, self), (-1, -1)

        """
        if it is turn of maximizing player, initial value will be (-) infinity. For each move in legal moves, create possible game snapshots and apply alphabeta.
        if it is turn of minimizing player, initial value will be (+) infinity. For each move in legal moves, create possible game snapshots and apply alphabeta.
        """
        if maximizing_player:
            max_ab = float("-inf"), legal_moves[0]
            
            for move in legal_moves:
                new_game = game.forecast_move(move)

                max_ab = max(max_ab, (self.alphabeta(new_game, depth - 1, alpha, beta, False)[0], move), key = lambda ab : ab[0])
                
                if max_ab[0] >= beta:
                    return max_ab
                    
                alpha = max(alpha, max_ab[0])
                
            return max_ab
        else:
            min_ab = float("inf"), legal_moves[0]
            
            for move in legal_moves:
                new_game = game.forecast_move(move)
                
                min_ab = min(min_ab, (self.alphabeta(new_game, depth - 1, alpha, beta, True)[0], move), key = lambda ab : ab[0])
                
                if min_ab[0] <= alpha:
                    return min_ab
                    
                beta = min(beta, min_ab[0])
                
            return min_ab          