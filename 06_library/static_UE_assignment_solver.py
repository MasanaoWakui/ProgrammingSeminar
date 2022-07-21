# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:06:40 2021

@author: 涌井 優尚


"""


import os
import numpy as np
import networkx as nx
import csv
import time
import pandas as pd
import copy



class prm:
    def __init__(self, dir_name, net_name, demand_plot=False):
        
        self.N, self.L         = self.read_net(dir_name, net_name)
        self.Q, self.zone_list = self.read_demand(dir_name, net_name)
        
        
    def read_net(self, dir_name, net_name):
        N = {} #ノードの情報
        L = {} #リンクの情報
        
        f = open("{}/{}_net.tntp".format(dir_name, net_name), "r")
        
        flag = False
        i = 0
        for line in f:# ここでファイルを一行ずつ読んでいる．一気に読まないので使用メモリが少ない（多分）
            if flag == True:
                link_parameter = line.split()# 一つの文字列を，スペースとか改行とかタブとかで分割する
    
                from_node = int(link_parameter[0])
                to_node   = int(link_parameter[1])
                ff_time   = float(link_parameter[4])
                capacity  = float(link_parameter[2])
        
                L[i+1] = {}
                L[i+1]["index"]     = i
                L[i+1]["from_node"] = from_node
                L[i+1]["to_node"]   = to_node
                L[i+1]["ff_time"]   = ff_time
                L[i+1]["capacity"]  = capacity
        
                if N.get(from_node):# Nの中に from_node があるなら
                    N[from_node]['outlink_list'] = N[from_node]['outlink_list'] + [i+1]
                else:
                    N[from_node] = {}
                    N[from_node]["index"] = from_node-1
                    N[from_node]['outlink_list'] = [i+1]
                    N[from_node]['inlink_list']  = []
            
                if N.get(to_node):# Nの中に to_node があるなら
                    N[to_node]['inlink_list'] = N[to_node]['inlink_list'] + [i+1]
                else:
                    N[to_node] = {}
                    N[to_node]["index"] = to_node-1
                    N[to_node]['outlink_list'] = []
                    N[to_node]['inlink_list']  = [i+1]
            
                i = i + 1
    
            if line[0] == "~":
                flag = True
                
        f.close()    
        return N, L
    
        
    def read_demand(self, dir_name, net_name):
        Q = {}
        zone_list = []

        f = open("../../ProgrammingSeminar/datafile/SiouxFalls_trips.tntp", "r")

        flag = False
        for line in f:# ここでファイルを一行ずつ読んでいる．一気に読まないので使用メモリが少ない（多分）
    
            if flag == True:
                dataline = line.split()# 一つの文字列を，スペースとか改行とかタブとかで分割する
                if dataline == []:
                    flag = False
                else:
                    lista = copy.deepcopy(dataline)
                    lista.reverse()
                    length = len(lista)
                    for i in range(length):
                        if lista[i] == ":":
                            j = length - 1 - i
                            del dataline[j]                    
            
                    for i in range(int(len(dataline)/2)):
                        dem = float(dataline[2*i+1].rstrip(";"))
                        Q[origin][int(dataline[2*i])] = dem
        
            if line[0] == "O":
                flag = True
                dataline = line.split()
                origin = int(dataline[1])
                zone_list = zone_list + [origin]
                Q[origin] = {}

        f.close()
        return Q, zone_list
        

        
        
        
class heap:
    def __init__(self, node_list):# ここでの node_list は（コスト，ノード番号）の２要素タプルを想定．
        self.h_list = []
        length = len(node_list)
        for i in range(length):
            self.heappush(node_list[i])
            # print(self.h_list)
    
    
    def swap(self, j, k):
        self.h_list[j], self.h_list[k] = self.h_list[k], self.h_list[j]
    
    
    def heappush(self, item):
        self.h_list.append(item)
        length = len(self.h_list)
        son = length - 1
        while True:
            par = (son - 1)//2
            if par < 0:
                break
            elif self.h_list[son][0] < self.h_list[par][0]:# コストのみ見て並べ替え
                self.swap(son, par)
                son = par
            else:
                break

    
    def heappop(self):
        tail = self.h_list.pop(-1)
        if self.h_list == []:
            return tail
        
        # 最後尾要素を先頭へ
        pop_item       = self.h_list[0]
        self.h_list[0] = tail
        
        length = len(self.h_list)
        
        # 上からヒープ作り直し
        par = 0
        while True:
            son = par*2 + 1
            if son >= length:
                break
            elif son+1 == length:
                if self.h_list[son][0] < self.h_list[par][0]:
                    self.swap(son, par)
                    par = son
                else:
                    break
            else:
                if self.h_list[son][0] < self.h_list[son+1][0]:
                    if self.h_list[son][0] < self.h_list[par][0]:
                        self.swap(son, par)
                        par = son
                    else:
                        break
                else:
                    if self.h_list[son+1][0] < self.h_list[par][0]:
                        self.swap(son+1, par)
                        par = son + 1
                    else:
                        break
            
        return pop_item
    
    

class model:
    def __init__(self, prm):
        
        self.prm = prm
        
        
    def solve(self, error_cut=10**(-4), conv_show=False):
        t_0 = np.array([self.prm.L[i+1]["ff_time"] for i in range(len(self.prm.L))])# 初期費用
        x_n = self.all_or_nothing(t_0)# 初期交通量
        z_n = self.objective(x_n)# 目的関数値
    
        n = 1
        while True:
        
            # Step 1. リンク費用更新
            t_n = self.BPR_vector(x_n)
        
            # Step 2. all-or-nothing 配分
            y_n = self.all_or_nothing(t_n)
        
            # Step 3. 直線探索
            d_n = y_n - x_n
            a_n = self.armijo_rule(x_n, d_n, t_n, z_n)
        
            # Step 4. 解の更新
            x_new = x_n + a_n*d_n
        
            # Step 5. 収束判定
            z_new = self.objective(x_new)

            aa = np.linalg.norm(x_new - x_n, ord=2)
            bb = np.linalg.norm(x_n, ord=2)
        
            if conv_show == True:
                print(n,"\t z:", z_new ,"\t , relative gap:",  aa/bb)
            
            if aa/bb < error_cut:
                t_new    = self.BPR_vector(x_new)
                solution = [x_new, t_new]
                print("x:\n", x_new)
                print("t:\n", t_new)
                break
            else:
                x_n = x_new
                z_n = z_new
                n   = n + 1
        
        return None
    
    
    
    # 目的関数とその勾配
    def objective(self, x):
        obj = 0
        for i in range(len(self.prm.L)):
            temp = self.prm.L[i+1]["ff_time"]*(x[i] + (x[i]**6)/(6*(self.prm.L[i+1]["capacity"])**5))
            obj = obj + temp
        
        return obj

    def BPR_vector(self, x):
        t = np.zeros(shape=len(self.prm.L))
        for i in range(len(self.prm.L)):
            t[i] = self.prm.L[i+1]["ff_time"]*(1 + (x[i]/self.prm.L[i+1]["capacity"])**5)
    
        return t
    
    
    
    
    # 配分計算
    def all_or_nothing(self, link_cost):
        y = np.zeros(shape=len(self.prm.L))
    
        # 全ての起点 i について順番に考える
        for i in self.prm.zone_list:
            pre_link = self.dijkstra(i, link_cost)[1]
        
            # すべての終点 j についてさらに順番に考える
            for j in self.prm.zone_list:
                link_set = []
                node = j
            
                # i から j への最短経路上のリンクを取得
                while True:
                    link = pre_link[node]
                    if link == 0:
                        break
                    else:
                        link_set = link_set + [link]
                        node = self.prm.L[pre_link[node]]["from_node"]
                
                # 最短経路上に需要を全て賦課
                q = self.prm.Q[i][j]
                for k in link_set:
                    y[self.prm.L[k]["index"]] = y[self.prm.L[k]["index"]] + q
        return y
    
    
    def dijkstra(self, origin, link_cost):
        INF = 10 ** 9
    
        num_node = len(self.prm.N)
        num_link = len(self.prm.L)
    
        # 最短経路上の直前リンクを記憶する辞書
        pre_link = {origin:0}
    
    
        spl  = np.ones(shape=num_node)*INF#shortest path length
        flag = np.zeros(shape=num_node)
    
        hq = heap([(0, origin)]) # (distance, node)
        spl[self.prm.N[origin]['index']]  = 0
    
        while hq.h_list:
            from_node = hq.heappop()[1] # 確定させるノードを pop
            
            if flag[self.prm.N[from_node]['index']] == 2:
                continue #探索済ならスキップ
            else:
                for l in self.prm.N[from_node]['outlink_list']:
                    to_node = self.prm.L[l]['to_node']
                    tmp_cost = spl[self.prm.N[from_node]['index']] + link_cost[self.prm.L[l]['index']]
            
                    if flag[self.prm.N[to_node]['index']] == 2:
                        continue #探索済ならスキップ
                    else:
                        if tmp_cost < spl[self.prm.N[to_node]['index']]:
                            spl[self.prm.N[to_node]['index']]  = tmp_cost
                            flag[self.prm.N[to_node]['index']] = 1 #探索中に変更
                            hq.heappush((tmp_cost, to_node)) #ヒープに追加
                            pre_link[to_node] = l
                
                flag[self.prm.N[from_node]['index']] = 2 #探索済に変更 
        
        return spl, pre_link
    
    
    
    
    #直線探索(アルミホのルール)
    def armijo_rule(self, x, d, gradient, z, max_step=1):
        al_alpha = 10**(-1)
        al_beta  = 0.5
        
        grad_z_d_prod = gradient@d
        
        l = 0
        while True:
            a = max_step*al_beta**l
            new_x = x + a*d
            leftside  = self.objective(new_x) - z
            rightside = al_alpha*a * grad_z_d_prod
            if leftside<=rightside:
                break
            l = l + 1

        return a