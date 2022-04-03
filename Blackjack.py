'''
PURPOSE:
    Simulate blackjack playing with n players against a dealer
AUTHOR:
    Malik Allahham
    
    
State:
    Components of the game that matter and affect the chance of winning:
        Our cards (Ace)
        Dealer card (that we can see)

Action:
    HIT
    STAND

Reward:
    (Doubled for double)
    Winning +1
    Losing -1
    Drawing +0
    
'''
import numpy as np
import pickle
import random
from time import time
        
class BlackJackSolution:

    def __init__(self, lr=0.001, exp_rate=0.4):
        self.player_Q_Values = {}  # key: [(player_value, shown_card, usable_ace)][action] = value
        # initialise Q values | [(4-21) x (1-10) x (True, False)] x (2, 1, 0) 600 in total
        for i in range(4, 22):
            for j in range(1, 11):
                for k in [True, False]:
                    self.player_Q_Values[(i, j, k)] = {}
                    for a in [2, 1, 0]:
                        if (i == 21) and (a == 0):
                            self.player_Q_Values[(i, j, k)][a] = 1
                        else:
                            self.player_Q_Values[(i, j, k)][a] = 0

        self.player_state_action = []
        self.state = (0, 0, False)  # initial state
        self.actions = [2, 1, 0]  # 2: DOUBLE 1: HIT  0: STAND
        self.end = False
        self.lr = lr
        self.exp_rate = exp_rate

    # give card
    @staticmethod
    def giveCard(deck):
        # 1 stands for ace, reshuffle after 70% of deck is cleared through shoe
        if len(deck) < (52 * 4 * 0.3):
            deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1]

        i = int(random.uniform(0, 1) * len(deck))
        card = deck[i]
        deck.pop(i)
        return card, deck

    def dealerPolicy(self, current_value, usable_ace, is_end, deck):
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True, deck
        # HIT17
        if current_value >= 17:
            return current_value, usable_ace, True, deck
        else:
            card, deck = self.giveCard(deck)
            if card == 1:
                if current_value <= 10:
                    return current_value + 11, True, False, deck
                return current_value + 1, usable_ace, False, deck
            else:
                return current_value + card, usable_ace, False, deck

    def chooseAction(self):
        # if current value < 9 always hit, if between 9 and 12 then double (can do more refinement later)
        current_value = self.state[0]
        if current_value < 9:
            return 1
        elif current_value < 12:
            return 2
        
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            v = float('-inf')
            action = 0
            for a in self.player_Q_Values[self.state]:
                if self.player_Q_Values[self.state][a] > v:
                    action = a
                    v = self.player_Q_Values[self.state][a]
        return action

    def playerNxtState(self, action, deck):
        current_value = self.state[0]
        show_card = self.state[1]
        usable_ace = self.state[2]

        if action == 1:
            card, deck = self.giveCard(deck)
            if card == 1:
                if current_value <= 10:
                    current_value += 11
                    usable_ace = True
                else:
                    current_value += 1
            else:
                current_value += card
        elif action == 0:
            self.end = True
            return (current_value, show_card, usable_ace)
        else:
            card, deck = self.giveCard(deck)
            if card == 1:
                if current_value <= 10:
                    current_value += 11
                    usable_ace = True
                else:
                    current_value += 1
            else:
                current_value += card
            self.end = True
            return (current_value, show_card, usable_ace)
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                self.end = True
                return (current_value, show_card, usable_ace)
            
        return (current_value, show_card, usable_ace)

    def winner(self, player_value, player_action, dealer_value):
        if player_action == 2:
            # player 2 | draw 0 | dealer -2
            winner = 0
            if player_value > 21:
                if dealer_value > 21:
                    winner = 0
                else:
                    winner = -2
            else:
                if dealer_value > 21:
                    winner = 2
                else:
                    if player_value < dealer_value:
                        winner = -2
                    elif player_value > dealer_value:
                        winner = 2
                    else:
                        # draw
                        winner = 0
            return winner
        
        else:
            # player 1 | draw 0 | dealer -1
            winner = 0
            if player_value > 21:
                if dealer_value > 21:
                    winner = 0
                else:
                    winner = -1
            else:
                if dealer_value > 21:
                    winner = 1
                else:
                    if player_value < dealer_value:
                        winner = -1
                    elif player_value > dealer_value:
                        winner = 1
                    else:
                        winner = 0
            return winner

    def _giveCredit(self, player_value, player_action, dealer_value, deck):
        
        reward = self.winner(player_value, player_action, dealer_value)
        # backpropagate reward
        for s in reversed(self.player_state_action):
            state, action = s[0], s[1]
            reward = self.player_Q_Values[state][action] + self.lr*(reward - self.player_Q_Values[state][action])
            self.player_Q_Values[state][action] = round(reward, 3)

    def reset(self):
        self.player_state_action = []
        self.state = (0, 0, False)
        self.end = False

    def deal2cards(self, deck, show=False):
        cards = [0, 0]
        value, usable_ace = 0, False
        cards[0], deck = self.giveCard(deck)
        cards[1], deck = self.giveCard(deck)
        if 1 in cards:
            value = sum(cards) + 10
            usable_ace = True
        else:
            value = sum(cards)
            usable_ace = False

        if show:
            return value, usable_ace, cards[0], deck
        else:
            return value, usable_ace, deck

    def play(self, deck, rounds=1000):
        for i in range(rounds):
            self.exp_rate = 0.4**(1 + ((i*5) / rounds))
            if len(deck) < (52 * 4 * 0.3):
                deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1]
            
            # give 2 cards to player and dealer
            dealer_value, d_usable_ace, show_card, deck = self.deal2cards(deck, show=True)
            player_value, p_usable_ace, deck = self.deal2cards(deck, show=False)

            self.state = (player_value, show_card, p_usable_ace)

            # judge winner after 2 cards
            if player_value == 21 or dealer_value == 21:
                # game end
                next
            else:
                while True:
                    action = self.chooseAction()
                    if self.state[0] >= 12:
                        state_action_pair = [self.state, action]
                        self.player_state_action.append(state_action_pair)
                    # update next state
                    self.state = self.playerNxtState(action, deck)
                    if self.end:
                        break

                        # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end, deck = self.dealerPolicy(dealer_value, d_usable_ace, is_end, deck)

                # judge winner
                # give reward and update Q value
                player_value = self.state[0]
                #print("player value {} | dealer value {}".format(player_value, dealer_value))
                self._giveCredit(player_value, action, dealer_value, deck)
  
            self.reset()

    def savePolicy(self, file="policy"):
        fw = open(file, 'wb')
        pickle.dump(self.player_Q_Values, fw)
        fw.close()

    def loadPolicy(self, file="policy"):
        fr = open(file, 'rb')
        self.player_Q_Values = pickle.load(fr)
        fr.close()

    def playWithDealer(self, deck, rounds=1000):
        self.reset()
        self.loadPolicy()
        self.exp_rate = 0
        runningCount = 0
        betSize = 10
        profit = 0
        
        result = np.zeros(3)  # player [win, draw, lose]
        for _ in range(rounds):
            if len(deck) < 70:
                runningCount = 0
                
            for i in deck:
                if i == 10 or i == 1:
                    runningCount += 1
                elif i <= 6 and i >= 2:
                    runningCount -= 1
                
            runningCount /= 4
    
            if runningCount < 1:
                betSize = 10
            elif runningCount >= 1 and runningCount < 2:
                betSize = 30
            elif runningCount >= 2 and runningCount < 3:
                betSize = 800
            elif runningCount >= 3:
                betSize = 1000
            
            dealer_value, d_usable_ace, show_card, deck = self.deal2cards(deck, show=True)
            player_value, p_usable_ace, deck = self.deal2cards(deck, show=False)

            self.state = (player_value, show_card, p_usable_ace)

            if player_value == 21 or dealer_value == 21:
                if player_value == dealer_value:
                    result[1] += 1
                elif player_value > dealer_value:
                    profit += 1.5 * betSize
                    result[0] += 1
                else:
                    profit -= 1 * betSize
                    result[2] += 1
            else:
                # player's turn
                while True:
                    action = self.chooseAction()
                    # update next state
                    self.state = self.playerNxtState(action, deck)
                    if self.end:
                        break

                # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end, deck = self.dealerPolicy(dealer_value, d_usable_ace, is_end, deck)

                # judge
                player_value = self.state[0]

                #print(player_value)
                #print("player value {} | dealer value {}".format(player_value, dealer_value))
                w = self.winner(player_value, action, dealer_value)
                if w == 1 and action == 2:
                    profit += 2 * betSize
                    result[0] += 1
                elif w == 1:
                    result[0] += 1
                    profit += betSize
                elif w == 0:
                    result[1] += 1
                else:
                    if action == 2:
                        result[2] += 1
                        profit -= 2 * betSize
                    else:
                        result[2] += 1
                        profit -= 1 * betSize
            self.reset()
            if _ % 100000 == 0:
                print(result)
        return result, profit, deck
        
if __name__ == "__main__":
    
    _rounds = 10000000
    
    # training
    
    b = BlackJackSolution()
    time0 = time()
    b.loadPolicy()
    deck = b.shuffle()
    
    #result = b.playWithDealer(rounds=_rounds)
    #print(result[0] / _rounds)
    
    #b.play(deck, _rounds)
    #b.savePolicy()
    
    
    b.play(deck, _rounds)
    b.savePolicy()
    print("Trained for", ((time() - time0) / 60) / 60, "hours")

    deck = b.shuffle()
    result, profit, deck = b.playWithDealer(deck, rounds= _rounds/100)
    print(result, (result[0] - result[2]))
    print(profit/_rounds)
   
    
   #This bad boi is fked up
   # Pls fix :(
    '''
    # Deck = shuffle function
    runningCount = 0
    profit = 0
    betSize = 10
    hands = 0
    totalBet = 0
    highBlackjacks = 0
    lowBlackjacks = 0
    # Loop
    while hands < 500000:
        playerVal = 0
        dealerVal = 0
        playerUseableAce = False
        dealerUseableAce = False
        
    #   If len(deck) < 33% of original; deck = shuffle, runningCount = 0
        if len(deck) < (52*4*0.3):
            deck = b.shuffle()
            runningCount = 0
            betSize = 10
            print("Shuffle")
            
    #   Draw two cards for player
        for l in range(0, 2):
            i = int(random.uniform(0, 1) * len(deck))
            playerVal += deck[i]
            if deck[i] == 11:
                playerUseableAce = True
                if playerVal == 22:
                    playerVal -= 10   
            deck.pop(i)
        
        if playerVal < 12:
            continue
        
    #   Draw card for dealer, pop in array
        i = int(random.uniform(0, 1) * len(deck))
        dealerVal += deck[i]
        if deck[i] == 11:
            dealerVal -= 10
            dealerUseableAce = True
        deck.pop(i)
        
    #   Check action for player
       # while playerVal < 12:
       #     i = int(random.uniform(0, 1) * len(deck))
       #     playerVal += deck[i]
       #     deck.pop(i)
            
        if playerVal > 21:
            if playerUseableAce:
                playerVal -= 10
                playerUseableAce = False

                
        b.state = (playerVal, dealerVal, playerUseableAce)
        
        totalBet += betSize
        
        if playerVal == 21 or dealerVal == 21:
                if playerVal == dealerVal:
                    profit += 0
                elif playerVal > dealerVal:
                    profit += 1.5 * betSize
                    if runningCount <= -1:
                        highBlackjacks += 1
                    else:
                        lowBlackjacks += 1
                    print("Blackjack!!!!")
                else:
                    #result[2] += 2.5
                    profit -= 1 * betSize
        
        else:
            # player's turn
            while True:
                action = b.chooseAction()
                #print(playerVal, dealerVal, playerUseableAce, action)
                # update next state
                b.state = b.playerNxtState(action)
                if b.end:
                    break
        
         
        # dealer's turn
            is_end = False
            while not is_end:
                dealerVal, dealerUseableAce, is_end = b.dealerPolicy(dealerVal, dealerUseableAce, is_end)

            # judge
            playerVal = b.state[0]
            # print("player value {} | dealer value {}".format(player_value, dealer_value))
            w = b.winner(playerVal, action, dealerVal)
            if w == 1 and action == 2:
                totalBet += betSize
                profit += 2 * betSize
            elif w == 1:
                profit += 1 * betSize
            elif w == 0:
                profit += 0
            else:
                if action == 2:
                    totalBet += betSize
                    #result[2] += 2
                    profit -= 2 * betSize
                else:
                    #result[2] += 1
                    profit -= 1 * betSize
        b.reset()
        
        #   Check bet size based on running count
        runningCount = 0
        for i in deck:
            if i >= 10:
                runningCount += 1
            elif i <= 6:
                runningCount -= 1
                
        runningCount /= 4
    
        if runningCount > -1:
            betSize = 10
        elif runningCount <= -1 and runningCount > -2:
            betSize = 50
        elif runningCount <= -2 and runningCount > -3:
            betSize = 700
        elif runningCount <= -3:
            betSize = 1000
            
        hands += 1
        #print(len(deck))
        #print(runningCount)
    print(lowBlackjacks)
    print(highBlackjacks)
    print(profit)
    print(totalBet)
    print(profit/totalBet)
    '''
    
    
    '''
    for k, v in b.player_Q_Values.items():
        actions = b.player_Q_Values.get(k)
        action = max(actions, key=actions.get)
        #action = max(actions.keys(), key=lambda k:actions[k])
        if action == 1:
            action = "HIT"
        elif action == 0:
            action = "STAND"
        elif action == 2:
            action = "DOUBLE"
        else:
            action = "BUG BOI"
        print(k, action)
'''
    # THIS IS FOR INPUTING AND TESTING USER DATA
    '''
    numDecks = input("How many decks are being played with? ")
    numDecks = int(numDecks)
    
    runningCount = 0
    print("Bet minimum")
    
    while(True):
        roundOver = False
        while(roundOver == False):
            playerVal = int(input("What is your current hand's value? "))
            dealerVal = int(input("What is the dealer's card? "))
            
            ace = input("Do you have an ace? ")
            if ace == "True":
                ace = True
            else:
                ace = False
            
            state = (playerVal, dealerVal, ace)
            
            if playerVal < 12:
                print("HIT")
            else:
                for k, v in b.player_Q_Values.items():
                    if state == k:
                        actions = b.player_Q_Values.get(k)
                        action = max(actions, key=actions.get)
                        #action = max(actions.keys(), key=lambda k:actions[k])
                        if action == 1:
                            action = "HIT"
                        elif action == 0:
                            action = "STAND"
                        elif action == 2:
                            action = "DOUBLE"
                        else:
                            action = "BUG BOI"
                        print(k, action)
            
            roundOver = input("Is the hand over? ")
            
            if roundOver == "True":
                roundOver = True
            else:
                roundOver = False
                
        runningCount *= numDecks
        
        addCard = 1
        while addCard != 0:
            addCard = int(input("Additional card on board? "))
            # Add additional cards on board to running count
            if addCard > 1 and addCard < 7:
                runningCount += 1
            elif addCard > 9:
                runningCount -= 1
        
        isShuffled = False
        
        if(isShuffled == False):
            isShuffled = input("Was the deck shuffled? ")
            if isShuffled == "True":
                runningCount = 0
            else:
                isShuffled = False
        
        runningCount /= numDecks
        if runningCount < 1:
            print("Bet minimum")
        elif runningCount >= 1 and runningCount < 2:
            print("Bet small")
        elif runningCount >=2 and runningCount < 3:
            print("Bet large")
        elif runningCount >= 3:
            print("Bet maximum")
        
       '''