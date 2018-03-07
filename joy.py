# coding=utf-8

from lainlib import *

#_____________________________________________________________/
# Main window configuration

window = tk.Tk()

window.title('Digimon Index :)')
window.geometry('900x600')

#_____________________________________________________________/
# Menu configuration





#window.mainloop()



#______________________________________________________________/



class Junct:

    def __init__(self, to_id, intensity):
        self.to_id = to_id
        self.intensity = intensity

# end class Junct




class Neura:

    def __init__(self, thresh, type):
        self.type = type
        self.thresh = thresh
        self.to = []
        self.sum = 0
        self.junct = []

    # end init

    def accumulate(self, value):
        self.sum += value
    # end

    def act(self):
        return self.thresh(self.sum)


    def show(self):
        print("type: " + str(self.type))
        print("junct to: ")
        for elem in self.to:
            print(elem.to_id)
        print()
    # end


# end class Neura




# class Artificial Neural Network
class ANN:

    def __init__(self, input_num, output_num):
        self.input_num = input_num
        self.output_num = output_num
        self.net = []
        self.queue = []

        base = input_num + output_num
        junct_output_possibility = 0.90
        junct_loop = 5
        initial_inter_num = 5

        # init input neura
        for i in range(input_num):
            self.net.append(Neura(log_sigmoid, -1))

            for t in range(junct_loop):
                chosed_id = base + math.floor(random()*5)
                if chosed_id not in self.net[i].junct:
                    self.net[i].to.append(Junct(chosed_id, random()))
                    self.net[i].junct.append(chosed_id)
                # end if
            # end for t
        # end for i

        # init output neura
        for j in range(output_num):
            #self.net.append(Neura(log_sigmoid, 1))
            self.net.append(Neura(lambda x: hard(x, 0.5, [0, 1]), 1))
        # end for j

        # init inter neura
        for k in range(initial_inter_num):
            self.net.append(Neura(log_sigmoid, 0))

            for t in range(junct_loop):
                chosed_id = input_num + math.floor(random()*(5+output_num))
                if chosed_id not in self.net[base+k].junct:
                    self.net[base+k].to.append(Junct(chosed_id, random()))
                    self.net[base+k].junct.append(chosed_id)
                # end if

                # make sure there's junctions juncting to output
                if random() <= junct_output_possibility:
                    output_id = input_num + math.floor(random()*(output_num+1))
                    if output_id not in self.net[base+k].junct:
                        self.net[base+k].to.append(Junct(output_id, random()))
                        self.net[base+k].junct.append(output_id)
                    # end
                # end if
            # end for t

        # end for k

    # end init




    def work(self, input):

        ans = []

        for i in range(self.input_num):
            for junct in self.net[i].to:
                self.net[junct.to_id].accumulate(input[i] * junct.intensity)
                if junct.to_id not in self.queue:
                    self.queue.append(junct.to_id)
                # end if
            # end junct
        # end for i

        for k in self.queue:
            for junct in self.net[k].to:
                self.net[junct.to_id].accumulate(self.net[k].act() * junct.intensity)
                if junct.to_id not in self.queue:
                    self.queue.append(junct.to_id)
                # end if
            # end for junct
            self.queue.remove(k)
        # end for j

        for k in range(self.output_num):
            ans.append(self.net[self.input_num+k].act())

        return ans




test = ANN(3, 10)

print(test.work([1, -2, -3]))


for elem in test.net:
    elem.show()












