import pandas as pd
import io  
import sys 

keyword = "x86_64"

class loginst:
    def __init__(self, logname, outname):
        try:
            self.files = open(logname,"rt",encoding='utf-8',errors='ignore')
            self.outfile = open(outname, "w+",encoding='utf-8',errors='ignore')
            self.flag = 0
            self.commit = ''
            self.date = ''
            self.author = ''
            self.content = ''
            self.tem_section = ''
        except:
            print('Open File Error! {}'.format(logname))
            
    def handle(self):
        line = self.files.readline()
        while line:
            if line.find("commit") == 0 :
                if self.flag == 3:
                    self.filtertofile(self.commit, self.date, self.author, self.content)
                    self.clearvar()
                    self.flag = 0
                self.commit = line
                self.flag = 1
            elif line.find("Author") == 0 :
                self.author = line
                self.flag = 2
            elif line.find("Date") == 0 :
                self.date = line
                self.flag = 3
            else:
                self.content += line
            line = self.files.readline()
        self.filtertofile(self.commit, self.date, self.author, self.content)
        self.files.close()
        self.outfile.close()

    def filtertofile(self, commit, date, author, content):
        self.tem_section = ''
        self.tem_section += commit
        self.tem_section += author
        self.tem_section += date
        self.tem_section += content
        if self.content.find(keyword) != -1 :
            self.outfile.write(self.tem_section)
            
    def clearvar(self):
        self.commit = ''
        self.author = ''
        self.date = ''
        self.content = ''

    def transform(self, commit, date, author, content):
        pass
        
        
def main():
#    train = pd.read_csv("./share/log.txt");
#    print(train.describe())
    logcls = loginst("./share/log.txt","out.txt")
    logcls.handle()

if __name__ == '__main__':
    #改变标准输出的默认编码 
#    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') 
    main()