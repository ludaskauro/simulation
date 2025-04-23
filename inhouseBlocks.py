from BaseBlocks import *
from SimulinkBlockClass import SimulinkBlock

class AccumCtr(SimulinkBlock):
    def __init__(self, name, raster, entry, Tol, I, DevMax, r, MxL, ValT, Ctr, Val):
        super().__init__(name, raster, [entry, Tol, I, DevMax, r, MxL, ValT], [Ctr, Val])

        self.label = 'AccumCtr'

        self.inputPorts = {entry:'entry', Tol:'Tol', I:'I', DevMax:'DevMax', r:'r', MxL:'MxL', ValT:'ValT'}
        self.outputPorts = {Ctr:'Ctr', Val:'Val'}

        self.addEntryCondition(EntryCondition(name='entry',entrySignal=entry))
        
        self.addBlock(Min(name='min1',signal1='delayedCtr',signal2=Tol,output='min1'))
        
        self.addBlock(Constant(name='Constant1',value=0))
        self.addBlock(Max(name='max1',signal1=I,signal2='Constant1',output='max1'))

        self.addBlock(Subtract(name='sub1',pos='max1',neg='min1',output='sub1'))

        self.addBlock(Min(name='min2',signal1='sub1',signal2=DevMax,output='min2'))

        self.addBlock(Add(name='add1',pos1='delayedCtr',pos2='min2',output='add1'))        

        self.addBlock(Constant(name='Constant3',value=0)) #this needs to be updated! -------------------------------------------------------
        self.addBlock(Max(name='max2',signal1='add1',signal2='Constant3',output='max2'))

        self.addBlock(Min(name='min3',signal1='max2',signal2='div1',output='min3'))

        self.addBlock(Constant(name='Constant4',value=0))
        self.addBlock(Switch(name='switchCtr',switched='Constant4',default='min3',condition=r,output='ctr1'))

        self.addBlock(Delay(name='delayCtr',signalIn='ctr1',signalOut='delayedCtr'))
        
        self.addBlock(Multiply(name='mul1',factor1='ctr1',factor2='ts',output=Ctr))

        self.addBlock(SampleTime(name='ts'))
        self.addBlock(Divide(name='div1',numerator=MxL,denominator='ts', output='div1'))

        self.addBlock(Add(name='add2',pos1='ts',pos2='delayVal',output='add2'))

        self.addBlock(Constant(name='Constant5',value=0))
        self.addBlock(Switch(name='switchVal',switched='Constant5',default='add2',condition=r,output='switchVal'))
        
        self.addBlock(Min(name='min4',signal1='switchVal',signal2=ValT,output='min4'))
        self.addBlock(Delay(name='delayVal',signalIn='min4',signalOut='delayVal'))

        self.addBlock(GreaterThanOrEqual(name='geq',first='switchVal',second=ValT,output=Val))

        self.compileBlock(printResult=False)

class RateLimiter(SimulinkBlock):
    def __init__(self, name, raster, signal, d, output):
        super().__init__(name, raster, [signal, d], [output])

        self.label = 'Rate Limiter'
        
        self.inputPorts = {signal:'In', d:'d'}

        self.addBlock(Max(name='max1',signal1=signal,signal2='sub1',output='max1'))
        
        self.addBlock(Min(name='min1',signal1='max1',signal2='add1',output=output))

        self.addBlock(Delay(name='delay1',signalIn=output,signalOut='delayedVal'))

        self.addBlock(SampleTime(name='ts'))

        self.addBlock(Multiply(name='mul1',factor1='ts',factor2=d,output='mul1'))

        self.addBlock(Multiply(name='mul2',factor1='ts',factor2=d,output='mul2'))

        self.addBlock(Subtract(name='sub1', pos='delayedVal',neg='mul1',output='sub1'))

        self.addBlock(Add(name='add1',pos1='mul2',pos2='delayedVal',output='add1'))

        self.compileBlock(printResult=False)

class _TimeAverageSubsystem(SimulinkBlock):
    def __init__(self, name, raster, entry, r, I, Ts, Time, O, Cmplt):
        super().__init__(name, raster, [entry, r, I, Ts, Time], [O, Cmplt])

        self.label = ''

        self.inputPorts = {entry:'entry', r:'r', I:'I', Ts:'Ts', Time:'Time'}
        self.outputPorts = {O:'O', Cmplt:'Cmplt'}

        self.addEntryCondition(EntryCondition(name='entry',entrySignal=entry))

        self.addBlock(Or(name='or1',signal1='delayedCmplt',signal2=r,output='reset'))

        self.addBlock(Constant(name='Constant1',value=1))

        self.addBlock(Switch(name='switchI',switched=I,default='addI',condition='reset',output='switchI'))
        self.addBlock(Switch(name='switchC',switched='Constant1',default='addC',condition='reset',output='switchC'))

        self.addBlock(Add(name='addI',pos1=I,pos2='delayedI',output='addI'))
        self.addBlock(Add(name='addC',pos1='Constant1',pos2='delayedC',output='addC'))

        self.addBlock(Constant(name='Constant2',value=1e30))
        self.addBlock(Min(name='minI',signal1='switchI',signal2='Constant2',output='minI'))
        self.addBlock(Min(name='minC',signal1='switchC',signal2='Constant2',output='minC'))

        self.addBlock(Delay(name='delayedI',signalIn='minI',signalOut='delayedI'))
        self.addBlock(Delay(name='delayedC',signalIn='minC',signalOut='delayedC'))

        self.addBlock(Divide(name='div',numerator='minI',denominator='minC',output=O))

        self.addBlock(Multiply(name='mul',factor1='minC',factor2=Ts,output='mul'))

        self.addBlock(GreaterThanOrEqual(name='geq',first='mul',second=Time,output=Cmplt))
        self.addBlock(Delay(name='delayedCmplt',signalIn=Cmplt,signalOut='delayedCmplt'))

        self.compileBlock(printResult=False)

class _TimeAverageRename(SimulinkBlock):
    def __init__(self, name, raster, entry, O, OH):
        super().__init__(name, raster, [entry,O], [OH])
        self.label = ''

        self.addEntryCondition(EntryCondition(name='entry',entrySignal=entry))

        self.addBlock(Constant(name='Constant1',value=1))
        self.addBlock(Multiply(name='mul',factor1=O,factor2='Constant1',output=OH))

        self.compileBlock(printResult=False)

class TimeAverage(SimulinkBlock):
    def __init__(self, name, raster, T, r, I, Time, Cmplt, OH, O):
        super().__init__(name, raster, [T, r, I, Time], [Cmplt, OH, O])

        self.label = 'Average'

        self.inputPorts = {T:'T', r:'r', I:'I', Time:'Time'}
        self.outputPorts = {Cmplt:'Cmplt', OH:'OH', O:'O'}

        self.addBlock(Or(name='or',signal1=T,signal2=r,output='entry1'))
        self.addBlock(And(name='and',signal1=T,signal2='Cmplt1',output=Cmplt))

        self.addBlock(SampleTime(name='ts'))

        self.addBlock(_TimeAverageSubsystem(name='subsystem',raster=raster,entry='entry1',I=I,r=r,Ts='ts',Time=Time,Cmplt='Cmplt1',O=O))

        self.addBlock(_TimeAverageRename(name='rename',raster=raster,entry=Cmplt,O=O,OH=OH))

        self.compileBlock(printResult=False)

class DiscreteDerivative(SimulinkBlock):
    def __init__(self, name, raster, signal, tc, output):
        super().__init__(name, raster, [signal, tc], [output])

        self.label = 'Discrete Derivative'

        self.inputPorts = {signal:'signal', tc:'tc'}

        self.addBlock(Delay(name='delay1',signalIn=signal, signalOut='delayedSignal'))
        self.addBlock(Subtract(name='sub1',pos=signal,neg='delayedSignal',output='sub1'))

        self.addBlock(SampleTime(name='ts'))
        self.addBlock(Constant(name='Constant1',value=1e-6))

        self.addBlock(Max(name='max1',signal1='ts',signal2='Constant1',output='max1'))
        self.addBlock(Divide(name='div1',numerator='sub1',denominator='max1',output='div1'))

        self.addBlock(Subtract(name='sub2',pos='div1',neg='delay2',output='sub2'))

        self.addBlock(Add(name='add1',pos1='ts',pos2=tc,output='add1'))

        self.addBlock(Max(name='max2',signal1='add1',signal2='Constant1',output='max2'))

        self.addBlock(Divide(name='div2',numerator='ts',denominator='max2',output='div2'))

        self.addBlock(Multiply(name='mul1',factor1='sub2',factor2='div2',output='mul1'))

        self.addBlock(Add(name='add2',pos1='delay2',pos2='mul1',output=output))

        self.addBlock(Delay(name='delay2',signalIn=output,signalOut='delay2'))

        self.compileBlock(printResult=False)

class DiscreteLowPassFilter(SimulinkBlock):
    def __init__(self, name, raster, signal, tc, output):
        super().__init__(name, raster, [signal, tc], [output])

        self.label = 'Discrete Low Pass Filter'

        self.inputPorts = {signal:'signal', tc:'tc'}

        self.addBlock(Subtract(name='sub1',pos=signal,neg='delay',output='sub1'))

        self.addBlock(SampleTime(name='ts'))

        self.addBlock(Max(name='max1',signal1='ts',signal2=tc,output='max1'))

        self.addBlock(Divide(name='div1',numerator='ts',denominator='max1',output='div1'))

        self.addBlock(Multiply(name='mul1',factor1='sub1',factor2='div1',output='mul1'))

        self.addBlock(Add(name='add1',pos1='delay',pos2='mul1',output=output))

        self.addBlock(Delay(name='delay',signalIn=output,signalOut='delay'))

        self.compileBlock(printResult=False)
