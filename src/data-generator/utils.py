
def readConf():
    conf = open('params.conf').readlines()
    params = [line.split(' = ') for line in conf if line[0] is not '#']
    params = [param for param in params if len(param) is 2]
    params = dict(zip([p[0].strip() for p in params],\
                      [p[1].strip() for p in params]))
    return params
