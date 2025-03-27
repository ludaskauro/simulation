from BaseBlocks import *
from SimulinkBlockClass import SimulinkBlock

class AccumCtr(SimulinkBlock):
    def __init__(self, name, entry, Tol, I, DevMax, r, MxL, ValT, Ctr, Val):
        super().__init__(name, [entry, Tol, I, DevMax, r, MxL, ValT], [Ctr, Val])

        self.node = [{'data':{'id':name,'label':'AccumCtr'}}] \
            + [{'data': {'id': self.name + '_' + i, 'parent': self.name,'label':i}} for i in self.inputs_list] \
            + [{'data': {'id': self.name + '_' + i, 'parent': self.name,'label':i}} for i in self.outputs_list]

        self.addEntryCondition(EntryCondition(name='entry',entrySignal=entry))
        
        self.addBlock(Min(name='min1',signal1='delayedCtr',signal2=Tol,output='min1'))
        
        self.addBlock(Constant(name='Constant1',value=0))
        self.addBlock(Max(name='max1',signal1=I,signal2='Constant1',output='max1'))

        self.addBlock(Subtract(name='sub1',pos='max1',neg='min1',output='sub1'))

        self.addBlock(Min(name='min2',signal1='sub1',signal2=DevMax,output='min2'))

        self.addBlock(Add(name='add1',pos1='delayedCtr',pos2='min2',output='add1'))

        self.addBlock(Constant(name='Constant3',value=0))
        self.addBlock(Max(name='max3',signal1='add1',signal2='Constant3',output='max3'))

        self.addBlock(Min(name='min3',signal1='max3',signal2='div1',output='min3'))

        self.addBlock(Constant(name='Constant4',value=0))
        self.addBlock(Switch(name='switchCtr',switched='Constant4',default='min3',condition=r,output='ctr1'))

        self.addBlock(Delay(name='delayCtr',signalIn='ctr1',signalOut='delayedCtr'))
        
        self.addBlock(Multiply(name='mul1',factor1='ctr1',factor2='ts',output=Ctr))

        self.addBlock(SampleTime(name='ts',value=0.04))
        self.addBlock(Divide(name='div1',numerator=MxL,denominator='ts', output='div1'))

        self.addBlock(Add(name='add2',pos1='ts',pos2='delayVal',output='add2'))

        self.addBlock(Constant(name='Constant5',value=0))
        self.addBlock(Switch(name='switchVal',switched='Constant5',default='add2',condition=r,output='switchVal'))
        
        self.addBlock(Min(name='min4',signal1='switchVal',signal2=ValT,output='min4'))
        self.addBlock(Delay(name='delayVal',signalIn='min4',signalOut='delayVal'))

        self.addBlock(GreaterThanOrEqual(name='geq',first='switchVal',second=ValT,output=Val))

        self.compileBlock()