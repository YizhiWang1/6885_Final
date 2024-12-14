import numpy as np
# import pygame
import random
import gymnasium as gym
from gymnasium import spaces

class EmotionWorld(gym.Env):
    def __init__(self, width=800, height=600, n_people=100):
        super(EmotionWorld, self).__init__()
        self.width = width # width of env
        self.height = height # height of env
        self.n_people = n_people # total number of people
        
        self.people = []
        for _ in range(n_people):
            person = {
                'position': np.array([random.randint(0, width), random.randint(0, height)], dtype=float),  # randomly initialize position
                'velocity': np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),  # randomly initialize velocity
                'personality': random.choice(['introvert', 'extrovert', 'neutral']), # randomly choose a personality
                'emotion': 'neutral', # initial emotion: 'neutral'
                'emotion_intensity': 0.0, # initial emotional intensity: have emotion=1.0, no emotion=0.0
                'emotion_duration': 0 # initialize duration of emotion 
            }
            self.people.append(person)
        
        # 'emotion type': spread_rate - possibility of spread, duration - how long, radius - how far
        self.emotions = {
            'joy': {'spread_rate': 0.8, 'duration': 100, 'radius': 50},
            'sadness': {'spread_rate': 0.8, 'duration': 120, 'radius': 80},
            'fear': {'spread_rate': 1.0, 'duration': 20, 'radius': 50},
            'disgust': {'spread_rate': 0.4, 'duration': 150, 'radius': 40},
            'anger': {'spread_rate': 0.6, 'duration': 50, 'radius': 120},
            'neutral': {'spread_rate': 0, 'duration': 0, 'radius': 0} # no emotion
        }
        # state space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(width // 10 * height // 10 * len(self.emotions),), dtype=np.float32
        )
        # action space
        self.num_emotions = len(['joy', 'sadness', 'fear', 'disgust', 'anger'])
        self.grid_size = 10
        self.action_space = spaces.Discrete(self.num_emotions * self.grid_size * self.grid_size)

        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # reset everyone's emtion
        for person in self.people: # reset to neutral emotion and zero emotion intensity
            person['emotion'] = 'neutral'
            person['emotion_intensity'] = 0.0
            person['emotion_duration'] = 0
        
        self.steps = 0 # reset step counter
        state = self._get_state()

        return state, {}
        
    def _update_positions(self):
        for person in self.people:
            # update position
            person['position'] += person['velocity'] 
            
            # check boundary: reverse the direction if out of boudary
            if person['position'][0] < 0 or person['position'][0] > self.width:
                person['velocity'][0] *= -1
            if person['position'][1] < 0 or person['position'][1] > self.height:
                person['velocity'][1] *= -1
    
    def _spread_emotion(self, emotion_type, target_pos):
        target_pos = np.array(target_pos)
        affected_count = 0 # initialize numbr of affected peope 
        
        for person in self.people:
            distance = np.linalg.norm(person['position'] - target_pos) # calculate distance to target position
            
            if distance <= self.emotions[emotion_type]['radius']: # if distance < radius
                spread_prob = self.emotions[emotion_type]['spread_rate'] * (1 - distance/self.emotions[emotion_type]['radius'])  
                # the possibility of spread will decrease(linear) based on the distance
                
                # adjust the spread rate based on personality
                if person['personality'] == 'extrovert':
                    spread_prob *= 1.2
                elif person['personality'] == 'introvert':
                    spread_prob *= 0.8
                
                if random.random() < spread_prob: # if emotion spread sucessfully
                    person['emotion'] = emotion_type  # change of emotion
                    person['emotion_intensity'] = 1.0  # emotion intensity = 1.0 
                    person['emotion_duration'] = self.emotions[emotion_type]['duration']
                    affected_count += 1 # affected people +1
        
        return affected_count

    def _spread_emotion_chain(self):
        chain_affected = 0  # initialize number of affected people

        for person in self.people:
            if person['emotion'] != 'neutral':  # emotional people spreading emotion
                emotion_type = person['emotion']
                for other in self.people:
                    if other['emotion'] == 'neutral':  # find neutral person (no emotion)
                        distance = np.linalg.norm(person['position'] - other['position'])
                    
                        if distance <= self.emotions[emotion_type]['radius']:
                            # spread probability
                            spread_prob = self.emotions[emotion_type]['spread_rate'] * (1 - distance/self.emotions[emotion_type]['radius'])
                         
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
            
                # if duration decrease to 0, set back to 'neutral' emotion
                if person['emotion_duration'] <= 0:
                    person['emotion'] = 'neutral'
                    person['emotion_intensity'] = 0.0

    def _get_state(self):
        # transfer environment to the input of neuronetwork
        state = np.zeros((self.width//10, self.height//10, len(self.emotions)))
    
        for person in self.people:
            x, y = person['position'] // 10
            
            # make sure the index is in range
            x = min(max(0, int(x)), state.shape[0] - 1)
            y = min(max(0, int(y)), state.shape[1] - 1)
            
            emotion_idx = list(self.emotions.keys()).index(person['emotion'])
            state[x, y, emotion_idx] += 1
        
        return state.flatten()
        

    
    def _is_episode_done(self):
        # ending requirement
        max_steps = 200  
        if getattr(self, 'steps', 0) >= max_steps:
            return True

        # mission is not complete if the requirement is not reached
        return False
    
    def step(self, action):
        """
        action: (emotion_type, target_position) 
        """
        # Decode the action
        emotion_idx = action // (self.grid_size * self.grid_size)
        position_idx = action % (self.grid_size * self.grid_size)
        emotion_type = list(self.emotions.keys())[emotion_idx]
        x = (position_idx // self.grid_size) * (self.width / self.grid_size)
        y = (position_idx % self.grid_size) * (self.height / self.grid_size)
        target_pos = (x, y)

        reward = 0
        
        # update people's position
        self._update_positions()

        # player trigger emotion spread
        directly_affected = self._spread_emotion(emotion_type, target_pos)
        reward += directly_affected  # number of people direcly affected by player

        # emotion spread chain
        reward = self._spread_emotion(emotion_type, target_pos)
        chain_affected = self._spread_emotion_chain()
        reward += chain_affected  # numbe of people affected by chain        
        
        # updateemotion spread
        self._update_emotions()
        
        # get new state
        new_state = self._get_state()
        
        # step counter + 1
        self.steps += 1

        # check if the episode is done
        done = self._is_episode_done()
        
        return new_state, reward, done, False, {}
