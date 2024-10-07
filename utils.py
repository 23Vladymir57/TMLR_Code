# path = "/home/volodimir/Bureau/ForLang/names.txt"
import torch
import numpy as np
import random
import pandas as pd


def lang_names(path):
    names = []
    extensions = []
    with open(path) as file:
        length = len(file.readlines())
    with open(path) as file:
        A = True
        h = 0
        while A:
            line = file.readline()
            h+=1
            A = not(line=='\n')
            if A:
                names.append(line[:-1])
        for j in range(length-h):
            line = file.readline()
            extensions.append(line[:-1])
    return (names,extensions)


def data_creator(name, path, ext, batch_size):
    data,_ = data_proces(path,name+ext[-1]+'.txt')
    l1 = get_max_length(data)
    input_length = len(alphabet_extractor(data))
    data,_ = data_proces(path,name+ext[0]+'.txt')
    l2 = get_max_length(data)
    data,_ = data_proces(path,name+ext[2]+'.txt')
    l3 = get_max_length(data)
    seq_leng = np.max([l1,l2,l3])

    train_data = Dataset(path,name+ext[-1]+'.txt',length = seq_leng)
    val_data   = Dataset(path,name+ext[0] +'.txt',length = seq_leng)
    test_data  = Dataset(path,name+ext[2] +'.txt',length = seq_leng)

    ### test zone


    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    return (train_loader, val_loader, test_loader, input_length, seq_leng)

def data_proces(path,name):
    data = []
    labels = []
    with open(path+name) as file:
        length = len(file.readlines())
    with open(path+name) as file:
        for j in range(length):
            line = file.readline()
            labeled_line = line.split()
            data.append(labeled_line[0])
            labels.append(labeled_line[1])
    return (data,labels)

def get_max_length(data):
    return np.max([len(x) for x in data])

def alphabet_extractor(data):
    alphabet = []
    for elem in data:
        for lettre in elem:
            alphabet.append(lettre)
    alphabet = set(alphabet)
    return sorted(list(alphabet))

def to_one_hot(self, x):
    x_one_hot = []
    for letter in x:
        x_one_hot.append(lettre_one_hot_encoder(self.one_hot,self.alphabet,letter))
    for _ in range(self.padding_length - len(x)):
        x_one_hot.append(np.zeros((len(self.alphabet),)))
    x_one_hot.append(len(x)*np.ones((len(self.alphabet),)))
    x_one_hot = np.array(x_one_hot)
    return torch.tensor(x_one_hot, dtype=torch.float32)

def lettre_one_hot_encoder(one_hot_list,alphabet,lettre):
    indx = integer_embeder(alphabet,lettre)
    return one_hot_list[indx]

def integer_embeder(alphabet,lettre):
    return alphabet.index(lettre)

def labeled_data_set(path, name, length):
    data = data_proces(path,name)
    alphabet = alphabet_extractor(data[0])
    target_alphabet = ['TRUE','FALSE']
    X = []
    y = []
    one_hot = []
    for j in range(len(alphabet)):
        one_hot.append(np.eye(len(alphabet))[:,j])
    
    if length == None:
        padding_length = get_max_length(data[0])
    else:
        padding_length = length
    
    for j in range(len(data[0])):
        x_one_hot = []
        x = data[0][j] 
        for letter in x:
            x_one_hot.append(lettre_one_hot_encoder(one_hot,alphabet,letter))
        for _ in range(padding_length - len(x)):
            x_one_hot.append(np.zeros((len(alphabet),)))
        x_one_hot.append(len(x)*np.ones((len(alphabet),)))
        X.append(x_one_hot)
        label = 1*(data[1][j]=='TRUE') + 0*(data[1][j]=='FALSE')
        y.append(label)
    return (X,y)

def data_creator2(name, path, ext):
    data,_ = data_proces(path,name+ext[-1]+'.txt')
    l1 = get_max_length(data)
    input_length = len(alphabet_extractor(data))
    data,_ = data_proces(path,name+ext[0]+'.txt')
    l2 = get_max_length(data)
    data,_ = data_proces(path,name+ext[2]+'.txt')
    l3 = get_max_length(data)
    seq_leng = np.max([l1,l2,l3])

    train_data = labeled_data_set(path,name+ext[-1]+'.txt',length = seq_leng)
    val_data   = labeled_data_set(path,name+ext[0]+'.txt' ,length = seq_leng)
    test_data  = labeled_data_set(path,name+ext[2]+'.txt' ,length = seq_leng)

    
    return (train_data, val_data, test_data, input_length, seq_leng)
    
def step(net,target,lr):
    ### normalisation of the direction and determening the norm of the gradient
    norm_2_acc = 0
    norm_inf_acc = []
    target_dist = 0
    norm_para  = 0
    k=0
    for parameters in net.parameters():
        gr = parameters.grad
        parameters = parameters.detach() 
        target_dist += ((torch.linalg.norm(parameters.flatten()-target[k].flatten())).item())**2
        parameters -= lr*(gr) # lr*(target[k]) + (1-lr)*(initial[k])
        parameters.requires_grad = True
        norm_2_acc += (torch.linalg.norm(gr.flatten(),ord=2).item())**2
        norm_para  += (torch.linalg.norm(parameters.flatten(),ord=2).item())**2
        norm_inf_acc.append(torch.linalg.norm(gr.flatten(),ord=float('inf')).item())
        k+=1
    target_dist = target_dist**(0.5)
    norm_2 = lr*norm_2_acc**(0.5)
    norm_para = norm_para**(0.5)
    norm_inf = max(norm_inf_acc)
    return (norm_2,norm_inf,target_dist, norm_para)

def stats(net,target,lr):
    ### this function returns the L2 and the L_\infty norm of the gradient
    norm_2_acc = 0
    norm_inf_acc = []
    target_dist = 0
    k=0
    for parameters in net.parameters():
        gr = parameters.grad
        parameters = parameters.detach() 
        target_dist += ((torch.linalg.norm(parameters.flatten()-target[k].flatten())).item())**2
        parameters -= lr*(gr) # lr*(target[k]) + (1-lr)*(initial[k])
        parameters.requires_grad = True
        norm_2_acc += (torch.linalg.norm(gr.flatten(),ord=2).item())**2
        norm_inf_acc.append(torch.linalg.norm(gr.flatten(),ord=float('inf')).item())
        k+=1
    target_dist = target_dist**(0.5)
    norm_2 = lr*norm_2_acc**(0.5)
    norm_inf = max(norm_inf_acc)
    return (norm_2,norm_inf,target_dist)

def fc_step(net,target,initial,lr, bl):
    ### normalisation of the direction and determening the norm of the gradient
    k=0
    ls_gr = []
    ls_st = []
    for parameters in net.parameters():
        ls_gr.append(parameters.grad.flatten())
        ls_st.append(parameters.flatten()-target[k].flatten())
        k+=1
    u = torch.cat(ls_gr)
    u_min = torch.min(torch.abs(u))
    u_max = torch.max(torch.abs(u))
    norm  = torch.linalg.norm(u)
    norm_zero = torch.linalg.norm(u,ord=0)
    v = torch.cat(ls_st)
    norm_st = torch.linalg.norm(v)

    u_min = nan_relu(u_min)
    u_max = nan_relu(u_max)
    norm  = nan_relu(norm)
    norm_zero = nan_relu(norm_zero)
    norm_st = nan_relu(norm_st)
    ### update step
    if bl:
        k=0
        for parameters in net.parameters():
            parameters = parameters.detach()
            parameters += lr*(target[k] - initial[k]) # lr*(target[k]) + (1-lr)*(initial[k])
            parameters.requires_grad = True
            k+=1
    
    return (norm,norm_zero,u_min,u_max,norm_st.detach())

def fc_partial_step(net,target,initial,lr, K, B,bl):
    ### determening the gradient norm
    k=0
    b=0
    ls_gr = []
    ls_st = []
    for parameters in net.parameters():
        if k<K:
            ls_gr.append(parameters.grad.flatten())
            ls_st.append(parameters.flatten()-target[k].flatten())
            k+=1
    u = torch.cat(ls_gr)
    u_min = torch.min(torch.abs(u))
    u_max = torch.max(torch.abs(u))
    print(u_max)
    norm  = torch.linalg.norm(u)
    norm_zero = torch.linalg.norm(u,ord=0)
    v = torch.cat(ls_st)
    norm_st = torch.linalg.norm(v)

    u_min = nan_relu(u_min)
    u_max = nan_relu(u_max)
    norm  = nan_relu(norm)
    norm_zero = nan_relu(norm_zero)
    norm_st = nan_relu(norm_st)

    ### update step
    if bl:
        k=0
        for parameters in net.parameters():
            if k<K:
                parameters = parameters.detach()
                parameters += lr*(target[k] - initial[k]) 
                parameters.requires_grad = True
            elif b< B and k>=K:
                gr = parameters.grad
                parameters = parameters.detach()
                parameters -= 0.001*gr 
                parameters.requires_grad = True
                b+=1
            k+=1
    return (norm,norm_zero,u_min,u_max,norm_st.detach())

def cf_step(net,target,initial,lr, bl):
    k=0
    for parameters in net.parameters():
        parameters = parameters.detach()
        parameters -= lr*(target[k] - initial[k]) 
        parameters.requires_grad = True
        k+=1
    
def cf_partial_step(net,target,initial,lr,K, bl):
    k=0
    for parameters in net.parameters():
        if k<K:
            parameters = parameters.detach()
            parameters -= lr*(target[k] - initial[k]) 
            parameters.requires_grad = True
            k+=1

def param_flater(net,reg_or_gred = 'grad'):
    """ prend en argument un réseau et retourne un vecteur de parametres ou du gradient avec l'information comment reconstituer les parametres
    """
    shapes = []
    length = 0
    params = net.state_dict()
    for key in params.keys():
        y=params[key].shape
        shapes.append(y)
        if len(y)==1:
            length += y[0]
        else:
            res = y[0]*y[1]
            length +=res
    fl_vec = torch.zeros((length))
    position  = 0
    for key in params.keys():
        shp = params[key].shape
        if len(shp)==1:
            fl_vec[position:position + shp[0]] = params[key]
            position = position + shp[0]
        else:
            for j in range(shp[1]):
                fl_vec[position:position + shp[0]] = params[key][:,j]
                position = position + shp[0]
    return (fl_vec, shapes)

def parame_deflater(fl_vec, shapes):
    """prend en argument un vecteur et une liste ordonée de tailles de matrices et vecteurs
       retourne une liste de 
    """
    position  = 0
    res_ls    = []
    for elem in shapes:
        curent = torch.zeros(elem)
        if len(elem)==1:
            curent = fl_vec[position:position + elem[0]]
            position += elem[0]
        else:
            for j in range(elem[1]):
                curent[:,j] = fl_vec[position:position + elem[0]]
                position += elem[0]
        res_ls.append(curent)
    return res_ls

def reparametrizer(param_lis, key_lis):
    """prend en paramtres une liste de tenseurs et une liste de nom de clé 
       retourne un dictionaire qu'on peut utiliser pour definir les parametres d'un réseau
    """
    out_dict = {}
    for j in range(len(param_lis)):
        out_dict[key_lis[j]] = param_lis[j]
    return out_dict

def H_approx(q_h):
    (previous, curent, lr) = q_h.get_stats()
    b = curent - previous
    x = previous
    b[b==0] = 10**(-7)
    x[x==0] = 10**(-7)
    H = b/(lr*x)
    return H

def Hgd(net,q_h):
    (fl_vec, shapes) = param_flater(net)
    H_inv = (H_approx(q_h))**(-1)
    fl_vec = H_inv*fl_vec
    out = parame_deflater(fl_vec, shapes)
    params = net.state_dict()
    key_lis = []
    for elem in params.keys():
        key_lis.append(elem)
    out_dict = reparametrizer(out, key_lis)
    k = 0
    for para in net.parameters():
        para.grad += out[k]
        k+=1
    
def nan_relu(tens):
    bl = torch.isnan(tens)
    if bl:
        return torch.zeros(tens.shape)
    else:
        return tens

def reset_grad(net):
    for parameters in net.parameters():
        parameters.grad *= 0

def gD_regular(net,target,q_h,training_set,lr,N,B = False):
    plot_loss  = []
    plot_norm_2   = []
    plot_norm_inf = []
    plot_dist = []
    plot_para = []
    mean_loss = 0
    A = True
    q_h.set_previous(q_h.get_stats()[1])
    for  (word, label) in random.sample(training_set,N):
        label = torch.tensor([label],dtype=torch.float32)
        net.reset_h()
        for lettre in word:
            lettre = lettre.to(torch.float32)
            outputs = net(lettre).to(torch.float32)
        loss = (-1)*( label*torch.log(outputs+10**(-7)) + (1-label)*torch.log((1-outputs)+10**(-7)) ) + 10**(-7)#criterion(outputs, label)
        mean_loss += (1/N)*loss
    mean_loss.backward()
    (fl_vec, shapes) = param_flater(net)
    q_h.set_curent(fl_vec)
    if B: ### trigers the Quasy Hessean acccelation 
        Hgd(net,q_h)
    (norm_2,norm_inf,target_dist, norm_para) = step(net,target,lr=lr)
    reset_grad(net)
    plot_norm_2.append(norm_2)
    plot_norm_inf.append(norm_inf)
    plot_loss.append(loss.detach().item())
    plot_dist.append(target_dist)
    plot_para.append(norm_para)
    mean_loss=0
    return (plot_loss,plot_norm_2,plot_norm_inf,plot_dist,plot_para,A)

def test(net,test_set):
    average = 0
    count = len(test_set)
    for  (word, label) in test_set:
        label = torch.tensor([label],dtype=torch.float32)
        net.reset_h()
        count
        for lettre in word:
            lettre = lettre.to(torch.float32)
            outputs = net(lettre).to(torch.float32)
        loss = (-1)*( label*torch.log(outputs+10**(-7)) + (1-label)*torch.log((1-outputs)+10**(-7)) ) + 10**(-7)
        print(loss)
        if loss < 10**(-7):
            average+=1
    print(average/count)
    
def rolling_avg(arr,window_size):


    
    window_size = 3
    
    # Convert array of integers to pandas series
    numbers_series = pd.Series(arr)
    
    # Get the window of series of
    # observations till the current time
    windows = numbers_series.expanding()
    
    # Create a series of moving averages of each window
    moving_averages = windows.mean()
    
    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()
    return np.array(moving_averages_list)
 
def random_strings(size,amount):
    res = []
    re  = []
    a = torch.tensor([1,0])
    b = torch.tensor([0,1])
    for j in range(amount):
        lis = []
        li  = 0
        for k in range(size):
            val = random.randint(0,1)
            lis.append(val*a + (1-val)*b)
            li+=val 
        re.append(li)
        res.append((lis,li%2))
    accepting = 0
    for elem in re:
        accepting+= elem
    print(accepting/amount)
    return res

def affected_params(net):    
    af = 0
    nonaf = 0
    for elem in net.parameters():
        vec = torch.floor(torch.log(torch.abs(elem.detach())))
        vec[vec<2] = 0
        vec[vec>=2]= 1
        af+= torch.count_nonzero(vec).numpy()
        nonaf+= torch.count_nonzero(1-vec).numpy()
    labels = ['Affected', 'Unaffected']
    sizes  = [af, nonaf]
    return (labels,sizes)

def train_stats(net,target, stats_data, optimizer, mini_batch, train_loader, val_loader, num_epochs, eps, lr, dtype, device):
    net.to(dtype=dtype,device=device)

    # Define loss and optimizer
    
    
    #train
    M = torch.eye(10)
    iteration = 0
    for _ in range(num_epochs):
        
        ## creating a batch
        batch_index = random.sample(list(range(len(train_loader[0]))),mini_batch)
        batch, labels = [], []
        for elem in batch_index:
            batch.append(train_loader[0][elem])
            labels.append(train_loader[1][elem])
        batch  = torch.tensor(batch)
        labels = torch.tensor(labels)

        optimizer.zero_grad()
        batch = batch.to(device=device)
        labels = labels.to(dtype=torch.long, device=device)
        
        # Forward pass
        outputs = net(batch).squeeze()
        loss = (-1)*( labels*torch.log(outputs+10**(-7)) + (1-labels)*torch.log((1-outputs)+10**(-7)) ) + torch.log(torch.tensor([1+10**(-7)])).to(dtype=torch.long, device=device)  #criterion(outputs, label)
        loss_plt = torch.sum(loss)/mini_batch

        # Backward and optimize
        loss_plt.backward()
        (norm_2,norm_inf,target_dist) = stats(net,target,lr)
        optimizer.step()

        
        stats_data.push('plot_loss',loss_plt.detach().item())
        stats_data.push('plot_norm',norm_inf)
        stats_data.push('plot_norm2',norm_2)
        stats_data.push('plot_dist',target_dist)
        
        iteration+=1
       
       
    print("The training is done")

def test_stat(net,test_loader,stats_data,dtype, device):
    net.to(dtype=dtype,device=device)

    ## creating a batch
    batch_index = list(range(len(test_loader[0])))
    batch, labels = [], []
    for elem in batch_index:
        batch.append(test_loader[0][elem])
        labels.append(test_loader[1][elem])
    batch  = torch.tensor(batch)
    labels = torch.tensor(labels)
        
  
    batch = batch.to(device=device)
    labels = labels.to(dtype=torch.long, device=device)
    
    # Forward pass
    
    outputs = net(batch).squeeze()
    loss = (-1)*( labels*torch.log(outputs+10**(-7)) + (1-labels)*torch.log((1-outputs)+10**(-7)) ) + torch.log(torch.tensor([1+10**(-7)])).to(dtype=torch.long, device=device)  #criterion(outputs, label)
    loss[loss<10**(-4)] = 0
    average = torch.count_nonzero(loss)/len(test_loader[0])

    stats_data.push('average_loss',average.detach().item())
    print(average)
    print("The testing is done")

