        
        
        
        
        
        ##################### FOR 6 TOTAL LAYERS #############################
        # wloss = np.dot (self.loss.reshape(1,Nodes[5]), WandB[8].reshape(Nodes[5],Nodes[4])) * DERIV[3]    
        # wloss = np.dot (wloss.reshape(1,Nodes[4]), WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2]    
        # wloss = np.dot (wloss.reshape(1,Nodes[3]), WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1]   
        # wloss = np.dot (wloss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DERIV[0]   
        # LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1)
        # LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1)   
        # newwloss = np.dot (self.loss.reshape(1,Nodes[5]), WandB[8].reshape(Nodes[5],Nodes[4])) * DERIV[3] # prob right
        # newwloss = np.dot (newwloss.reshape(1,Nodes[4]),WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2] # prob right        
        # newwloss = np.dot (newwloss.reshape(1,Nodes[3]),WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1] # prob right
        # LayerWandB[2]= newwloss.reshape (Nodes[2],1) * ACTIV[0].reshape(1, Nodes[1]) * 1/(k+1) # should be 356,533 not 122,533
        # LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[2],1) * 1/(k+1) 
        # nloss = np.dot (self.loss.reshape(1,Nodes[5]), WandB[8].reshape(Nodes[5],Nodes[4])) * DERIV[3]
        # nloss = np.dot (nloss.reshape(1,Nodes[4]),WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2]
        # LayerWandB[4]= nloss.reshape (Nodes[3],1) * ACTIV[1].reshape(1, Nodes[2]) * 1/(k+1)  #w2 = 122,356
        # LayerWandB[5] = np.sum(LayerWandB[4],1).reshape(Nodes[3],1) * 1/(k+1) # 356,1      # 3 # 2 ,1 
        # otherloss = np.dot (self.loss.reshape(1,Nodes[5]), WandB[8].reshape(Nodes[5],Nodes[4])) * DERIV[3]
        # LayerWandB[6] = otherloss.reshape(Nodes[4],1) * ACTIV[2].reshape(1, Nodes[3]) * 1/(k+1)  #w2 = 122,356
        # LayerWandB[7] = np.sum(LayerWandB[6],1).reshape(Nodes[4],1) * 1/(k+1) 
        # w1loss  = self.loss.reshape(Nodes[5],1) *  ACTIV[3].reshape(1,Nodes[4])
        # LayerWandB[8] = w1loss.reshape(Nodes[5],Nodes[4]) * 1/(k+1) 
        # LayerWandB[9] = np.sum(LayerWandB[8],1).reshape(Nodes[5],1) * 1/(k+1)  
        #######################################################################

        ##################### FOR 5 TOTAL LAYERS ############################
        # wloss = np.dot (self.loss.reshape(1,Nodes[4]), WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2]    
        # wloss = np.dot (wloss.reshape(1,Nodes[3]), WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1]    
        # wloss = np.dot (wloss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DERIV[0]   
        # LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1)
        # LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1)         
        # newwloss = np.dot (self.loss.reshape(1,Nodes[4]),WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2] # prob right
        # newwloss = np.dot (newwloss.reshape(1,Nodes[3]),WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1] # prob right
        
        # LayerWandB[2]= newwloss.reshape (Nodes[2],1) * ACTIV[0].reshape(1, Nodes[1]) * 1/(k+1) # should be 356,533 not 122,533
        # LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[2],1) * 1/(k+1) 
        # nloss = np.dot (self.loss.reshape(1,Nodes[4]), WandB[6].reshape(Nodes[4],Nodes[3])) * DERIV[2]    
        # LayerWandB[4]= nloss.reshape (Nodes[3],1) * ACTIV[1].reshape(1, Nodes[2]) * 1/(k+1)  #w2 = 122,356
        # LayerWandB[5] = np.sum(LayerWandB[4],1).reshape(Nodes[3],1) * 1/(k+1) # 356,1      # 3 # 2 ,1 
        # w1loss  = self.loss.reshape(Nodes[4],1) *  ACTIV[2].reshape(1,Nodes[3])
        # LayerWandB[6] = w1loss.reshape(Nodes[4],Nodes[3]) * 1/(k+1) 
        # LayerWandB[7] = np.sum(LayerWandB[6],1).reshape(Nodes[4],1) * 1/(k+1)  
        #######################################################################

        ##################### FOR 4 TOTAL LAYERS ############################   
        # wloss = np.dot (self.loss.reshape(1,Nodes[3]), WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1]     # 1, 356
        # wloss = np.dot (wloss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DERIV[0]     # 1,533
        # LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1)  # 533,784
        # LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1) # 533,1
        # newwloss = np.dot (self.loss.reshape(1,Nodes[3]),WandB[4].reshape(Nodes[3],Nodes[2])) * DERIV[1]
        # LayerWandB[2]= newwloss.reshape (Nodes[2],1) * ACTIV[0].reshape(1, Nodes[1]) * 1/(k+1) # 356,533 # 4
        # LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[2],1) * 1/(k+1) # 356,1      # 3 # 2 ,1 
        # w1loss  = self.loss.reshape(Nodes[3],1) *  ACTIV[1].reshape(1,Nodes[2])
        # LayerWandB[4] = w1loss.reshape(Nodes[3],Nodes[2]) * 1/(k+1) 
        # LayerWandB[5] = np.sum(LayerWandB[4],1).reshape(Nodes[3],1) * 1/(k+1)  
        #######################################################################

        ##################### FOR 3 TOTAL LAYERS ############################
        # wloss = np.dot (self.loss.reshape(1,Nodes[2]), WandB[2].reshape(Nodes[2],Nodes[1])) * DERIV[0]    
        # LayerWandB[0] = wloss.reshape(Nodes[1],1)  * self.scaledarray.reshape(1,Nodes[0]) * 1/(k+1) # 533,784
        # LayerWandB[1] = np.sum(LayerWandB[0],1).reshape(Nodes[1],1) * 1/(k+1) # 533,1
        # w1loss  = self.loss.reshape(Nodes[2],1) *  ACTIV[0].reshape(1,Nodes[1]) # 3N = 0 , 4N = 2
        # LayerWandB[2] = w1loss.reshape(Nodes[2],Nodes[1]) * 1/(k+1)  # 10,356
        # LayerWandB[3] = np.sum(LayerWandB[2],1).reshape(Nodes[2],1) * 1/(k+1)   # 10,1
        ###############################################################