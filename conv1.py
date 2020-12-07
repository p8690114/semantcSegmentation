import os
import pdb 
#pdb.set_trace()
path = os.getcwd()+"/ss3"
output = os.getcwd()+"/output3"
fileDirs=os.listdir(path)
for fileDir in fileDirs:
    file=path+"/"+fileDir+"/label.png"
    if(os.path.exists(file)):
        # 輸出的文件直接以上層文件夾命名
        end= len(fileDir);
        fileName=fileDir[:end-5]
        os.rename(file,output+"/"+fileName+".png")
