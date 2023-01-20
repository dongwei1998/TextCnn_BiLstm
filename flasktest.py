# coding=utf-8
# =============================================
# @Time      : 2022-04-21 16:40
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import json,time,requests

def element_extract():
    url = f"http://10.19.234.179:5557/api/v1/classification"
    demo_text ={
        'text':'''
            本院在审理原告中国电信股份有限公司英德分公司九龙营销服务中心诉被告李伯尚电信服务合同纠纷一案中，原告中国电信股份有限公司英德分公司九龙营销服务中心于2013年9月10日向本院提出撤诉申请。
            本院认为，原告有权依法处分自己的民事权利和诉讼权利。原告的申请，符合法律的规定，本院予以准许。依照《中华人民共和国民事诉讼法》第十三条、第一百五十四条的规定，裁定如下：
        '''
    }

    headers = {
        'Content-Type': 'application/json'
    }
    start = time.time()
    result = requests.post(url=url, json=demo_text,headers=headers)
    end = time.time()
    if result.status_code == 200:
        obj = json.loads(result.text)
        print(obj)
    else:
        print(result)
    print('Running time: %s Seconds' % (end - start))


element_extract()

