from game import controlled_run

import numpy as np
from game import DO_NOTHING
from game import JUMP

games_count=0
total_number_of_games =10


import numpy as np
from game import DO_NOTHING
from game import JUMP

games_count=0
total_number_of_games =10


import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Dense(1,input_dim=1,activation='sigmoid'))
model.add(Dense(2,activation='softmax'))
model.compile(Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])

x_train = np.array([])
y_train = np.array([])


# flaw in game design, it should be based on whether or not you make it past an opponent
class Wrapper(object):
    
    
    def __init__(self):
        controlled_run(self,0)
        
    def control(self,values):
        global x_train
        global y_train
        
        print(values)
        
        if(values['closest_enemy'] == -1):
            return DO_NOTHING
        
        if(values['old_closest_enemy']!=-1):
            if(values['score_increased']==1):
                x_train = np.append(x_train,[values['old_closest_enemy']/1000])
                y_train = np.append(y_train,[values['action']])
                
        prediction=model.predict_classes(np.array([[values['closest_enemy']]])/1000)
            
        
        return prediction
    
    def gameover(self,score):
        global games_count
        global x_train
        global y_train
        
        print(x_train)
        print(y_train)
        
        if(games_count >0):
            y_train_cat = to_categorical(y_train,num_classes=2)
            model.fit(x_train,y_train_cat,epochs=50, verbose =1,shuffle=1)
            
        games_count+=1
        
        if games_count>=total_number_of_games: 
            return
        controlled_run(self,games_count)

w=Wrapper()