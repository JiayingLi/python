# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:39:26 2019

@author: jiaying
"""

import re
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging, logging.config
import threading
import queue


queue1 = queue.Queue()
url_queue = queue.Queue()
url_queue2 = queue.Queue()

#logging.config.fileConfig("log.conf")
#log = logging.getLogger()
#补充的首页上没有跳转的分类
supplement_url = ['list.jd.com/list.html?cat=4053,17338|音乐||0']

#初始化匹配规则
re1 = re.compile(r'(\d+\-\d+\-*\d*)\|(.*)\|\|')
re2 = re.compile(r'(.*list\.jd\.com.*)\|(.*)\|\|')
re3 = re.compile(r'(coll\.jd\.com.*)\|(.*)\|\|')

category = {}

class ThreadUrl(threading.Thread):
    def __init__(self,queue1,url_queue):
        threading.Thread.__init__(self)
        self.queue1 = queue1
        self.url_queue = url_queue
 
    def run(self):
        while True:
            url = self.queue1.get()
            res = requests.get(url)
            bs = BeautifulSoup(res.content,'lxml')
            self.url_queue.put(bs)#将hosts中的页面传给out_queue
            self.queue1.task_done()

def getJDCategory():
    """
    从jd的分类目录拉取分类信息,返回json格式的数据
    """
    url=r"https://dc.3.cn/category/get"
    head = {
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9',
            'referer': 'https://www.jd.com/allSort.aspxjdf/1.0.0/ui/elevator/1.0.0/elevator',
            'user-agent': 'Chrome/65.0.3325.181'  
    }
    param = {'callback': 'getCategoryCallback'}
    res=requests.get(url,params=param,headers=head)
    #转换为json文件格式
    content = re.match(b'getCategoryCallback\((.*)\)$',res.content).group(1)
    return content

def match_url(url):
    """
    匹配返回有效url,若无效则返回空字符串
    """
    result = ''
    if re1.search(url):
        param = ','.join(re1.search(url).group(1).split('-'))
        result = r'https://list.jd.com/list.html?cat='+param
    if re2.search(url):
        result = r'https://'+re2.search(url).group(1)
    #if re3.search(t):
    #   url = r'https://'+re3.search(t).group(1)
    return result

class getCategoryInfo(threading.Thread):
    def __init__(self, queue1, url_queue):
        threading.Thread.__init__(self)
        self.queue1 = queue1
        self.url_queue = url_queue

    def run(self):
        while True:
            bs = self.url_queue.get()
            
            #获取一级分类名
            title = [x.text for x in bs.select(".crumbs-nav-item a")]
            if not title:
                self.url_queue.task_done()
                continue
            t1 = title[0]
            
            #获取二三级分类名
            items = [x.text for x in bs.select(".crumbs-nav-item .trigger")]
            t2 = items[0]
            
            if t1 not in category:
                    category[t1] = {}
                    print(f"正在获取分类：{t1}.")
            if t2 not in category[t1]:
                category[t1][t2] = []
            try:
                #若三级分类不在表里则添加
                t3 = items[1]
                if t3 not in category[t1][t2]:
                    category[t1][t2].append(t3)
                
                #在当前url下拉菜单中补充缺失的三级分类
                bs1 = bs.select(".crumbs-nav-item")[2]
                DropDown_title3 = [t.text for t in bs1.find_all('a')]
                for t3 in DropDown_title3:
                    if t3 not in category[t1][t2]:
                        category[t1][t2].append(t3)
                
                #若其他二级分类未在表里则添加进url列表
                bs2 = bs.select(".crumbs-nav-item")[1]
                DropDown_title2 = [t for t in bs2.find_all('a')]
                for title2 in DropDown_title2:
                    if title2.get('title') and title2.get('title') not in category[t1]:
                        category[t1][title2.get('title')] = []
                        if re.search('https.*cat',title2.get('href')):
                            url = match_url(title2.get('href').strip('https://')+'|补充||0')
                            if url:
                                self.queue1.put(url)
                        elif re.search('cat',title2.get('href')):
                            url = match_url('list.jd.com'+title2.get('href')+'|补充||0')
                            if url:
                                self.queue1.put(url)
                
            except IndexError:
                #横条中没有三级分类则从下面的表格中获取
                title3 = bs.select("#J_selectorCategory a")
                title3 = [x.get("title") for x in title3 if title3]
                category[t1][t2] += [x for x in title3 if x and x not in category[t1][t2]]
                
                #若其他二级分类未在表里则添加进url列表
                bs2 = bs.select(".crumbs-nav-item a")
                for title2 in bs2:
                    if title2.get('title') and title2.get('title') not in category[t1]:
                        category[t1][title2.get('title')] = []
                        url = match_url('list.jd.com'+title2.get('href')+'|补充||0')
                        if url:
                            self.queue1.put(url)
            self.url_queue.task_done()

def multiThreadGetinfo():
    start = time.time()
    #向京东商品首页发送请求
    content = getJDCategory()

    #使用BeautifulSoup解析数据
    soup = BeautifulSoup(content, "html.parser")
    j = json.loads(soup.text)
    
    #获取首页里的所有三级分类的url
    third_title = []
    third_title+=supplement_url
    for first in j['data']:
        for f in first['s']:
            for second in f['s']:
                for third in second['s']:
                    third_title.append(third['n'])
    
    print("正在获取分类信息....")
    #获取三级分类对应的一二三级分类信息

    for i in range(10):
        t1 = ThreadUrl(queue1, url_queue)
        t1.setDaemon(True)#设置为守护线程
        t1.start()
    
    for t in third_title:
        #匹配获取有效url
        url = match_url(t)
        
        #跳转进入三级分类页面，获取一二三级分类对应关系
        if url:
            queue1.put(url)
    
    for i in range(10):
        t2 = getCategoryInfo(queue1,url_queue)
        t2.setDaemon(True)#设置为守护线程
        t2.start()
    
    queue1.join()
    url_queue.join()

    end = time.time()
    print(f"分类获取完成！耗时：{round(end-start,2)}s")
    print("正在输出格式化数据,每个分类按行存储...")
    output = []
    for f in category:
        for s in category[f]:
            for t in category[f][s]:
                output.append([f,s,t])
    
    return output
    #储存为csv文件
#    df = pd.DataFrame(output)
#    df.to_csv('测试.csv', encoding='utf_8_sig',header = ['一级分类','二级分类','三级分类'],index=False)


