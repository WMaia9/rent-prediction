import math
import numpy as np

def _dist(lata, lona, latb, lonb):
    """
    Calcula a distância geoespacial entre 2 pontos próximos
        
    """
    lata = lata * math.pi / 180
    lona = lona * math.pi / 180
    latb = latb * math.pi / 180
    lonb = lonb * math.pi / 180
    R = 6372795.4775979994
    dis = int(R * math.acos(math.sin(lata) * math.sin(latb) + math.cos(lata) * math.cos(latb) * math.cos(lona-lonb)))
    return dis

distance = np.vectorize(_dist)
""" 
vetoraliza a função de distância
"""
