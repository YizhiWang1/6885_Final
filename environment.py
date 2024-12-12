import numpy as np
import pygame
import random

class EmotionWorld:
    def __init__(self, width=800, height=600, n_people=100):
        self.width = width # width of env
        self.height = height # height of ev
        self.n_people = n_people # total number of people
        
        # 
        self.people = []
        for _ in range(n_people):
            person = {
                'position': np.array([random.randint(0, width), random.randint(0, height)]), # randomly initialize position
                'velocity': np.array([random.uniform(-1, 1), random.uniform(-1, 1)]), # randomly initialize velocity
                'personality': random.choice(['introvert', 'extrovert', 'neutral']), # randomly choose a personality
                'emotion': 'neutral', # initial emotion
                'emotion_intensity': 0.0, #initial emotional intenity, have emotion=1.0 no emotion=0.0
                'emotion_duration': 0 # intianlize duration of emotion
            }
            self.people.append(person)
        
        # 'emotion type': rate - possibility of spread, duration - how long, radius - how far
        self.emotions = {
            'joy': {'spread_rate': 0.8, 'duration': 100, 'radius': 50},
            'sadness': {'spread_rate': 0.8, 'duration': 120, 'radius': 40},
            'fear': {'spread_rate': 1.0, 'duration': 20, 'radius': 30},
            'disgust': {'spread_rate': 0.4, 'duration': 150, 'radius': 20},
            'anger': {'spread_rate': 0.6, 'duration': 50, 'radius': 80},
            'neutral': {'spread_rate': 0, 'duration': 0, 'radius': 0} # no emotion
        }
        
        self.reset()
    
    def reset(self):
        # reset everyone's emtion
        for person in self.people: # reset to neutral emotion and no emotion intensity
            person['emotion'] = 'neutral'
            person['emotion_intensity'] = 0.0
        
        # restart
        return self._get_state()
        
    def _update_positions(self):
        # logic of moving
        for person in self.people:
            # update position
            person['position'] += person['velocity'] 
            
            # check boundary：reverse the direction if out of boudary
            if person['position'][0] < 0 or person['position'][0] > self.width:
                person['velocity'][0] *= -1
            if person['position'][1] < 0 or person['position'][1] > self.height:
                person['velocity'][1] *= -1
    
    def _spread_emotion(self, emotion_type, target_pos):
        target_pos = np.array(target_pos)
        affected_count = 0 #initialize numbr of peope affeced
        
        for person in self.people:
            distance = np.linalg.norm(person['position'] - target_pos) #calculating distance to target position
            
            if distance <= self.emotions[emotion_type]['radius']: # if distance < radius
                spread_prob = self.emotions[emotion_type]['spread_rate'] * \
                            (1 - distance/self.emotions[emotion_type]['radius'])  # calculate possibility of spread, will decrease(linear) based on the distance
                
                # effect of personality : adjust the spread rate based on personality
                if person['personality'] == 'extrovert':
                    spread_prob *= 1.2
                elif person['personality'] == 'introvert':
                    spread_prob *= 0.8
                
                if random.random() < spread_prob: # comparing spread rate with random number to see if emotion spread sucessfully
                    person['emotion'] = emotion_type  # change of emotion
                    person['emotion_intensity'] = 1.0  # emotion intensity = 1.0 
                    person['emotion_duration'] = self.emotions[emotion_type]['duration']
                    affected_count += 1 # affected people+1
        
        return affected_count

    def _spread_emotion_chain(self):
        chain_affected = 0 # initialize number of affected people

        for person in self.people:
            if person['emotion'] != 'neutral':  # emotional people spreadig emotion
                emotion_type = person['emotion']
                for other in self.people:
                    if other['emotion'] == 'neutral':  #find neutral person(no emotion)
                        distance = np.linalg.norm(person['position'] - other['position'])
                    
                        if distance <= self.emotions[emotion_type]['radius']:
                            # spread probabilit
                            spread_prob = self.emotions[emotion_type]['spread_rate'] * \
                            (1 - distance/self.emotions[emotion_type]['radius'])
                         
                            # effects of personality
                            if other['personality'] == 'extrovert':
                                spread_prob *= 1.2
                            elif other['personality'] == 'introvert':
                                spread_prob *= 0.8
                        
                            # emotion is spread
                            if random.random() < spread_prob:
                                other['emotion'] = emotion_type
                                other['emotion_intensity'] = 1.0
                                other['emotion_duration'] = self.emotions[emotion_type]['duration']
                                chain_affected += 1

        return chain_affected

    def _update_emotions(self):
        # update emotion duration
        for person in self.people:
            # if the emotion is not 'neutral', decrease the emotoin duration 
            if person['emotion'] != 'neutral':
                person['emotion_duration'] -= 1
            
                # if duration decrease to 0，set back to 'neutral' emotion
                if person['emotion_duration'] <= 0:
                    person['emotion'] = 'neutral'
                    person['emotion_intensity'] = 0.0

    def _get_state(self):
        # transfer environment to the input of neuronetwork
        state = np.zeros((self.width//10, self.height//10, len(self.emotions)))
        
        # 10 pixels = 1 grid, accumulating numbr of different emotion of each grid, then flatting them to an array
        for person in self.people:
            x, y = person['position'] // 10
            emotion_idx = list(self.emotions.keys()).index(person['emotion'])
            state[int(x), int(y), emotion_idx] += 1
        
        return state.flatten()
    
    def _is_episode_done(self):
        # ending requirement:
        max_steps = 200  
        if getattr(self, 'steps', 0) >= max_steps:
            return True

        # mission is not complete if the requirement is not reached
        return False
    
    def step(self, action):
        """
        action: (emotion_type, target_position) emotion type  objective position
        """
        emotion_type, target_pos = action
        reward = 0
        
        # update people's position
        self._update_positions()

        # player trigger emotion spred
        directly_affected = self._spread_emotion(emotion_type, target_pos)
        reward += directly_affected  # number of people direcly affected by player

        # emotion spread chain
        chain_affected = self._spread_emotion_chain()
        reward += chain_affected  # numbe of people affected by chain   
        
        # update emotion spread
        self._update_emotions()
        
        # get new state
        new_state = self._get_state()
        
        # check if the episode is done
        done = self._is_episode_done()
        
        return new_state, reward, done

# Goal: maximize the number of people affected by emotions through actions
