import glob

apples = []
for filename in glob.glob("L:/Projects/Apples_vs_Tomatoes/Dataset/Apples/*.png"):
  apples.append(filename)

with open('L:/Projects/Apples_vs_Tomatoes/Dataset/apples.txt', 'w') as f:
   for item in apples:
        f.write("{}\n".format(item))
        
tomatoes = []
for filename in glob.glob("L:/Projects/Apples_vs_Tomatoes/Dataset/Tomatoes/*.png"):
  tomatoes.append(filename)

with open('L:/Projects/Apples_vs_Tomatoes/Dataset/tomatoes.txt', 'w') as f:
   for item in tomatoes:
        f.write("{}\n".format(item))