## Introduction

Starting with the Kepler architecture Nvidia has been moving some control logic off of the chip and into kernel instructions which are determined by the assembler.  This makes sense since it cuts down on die space and power usage, plus the assembler has access to the whole program and can make more globally optimal decisions about things like scheduling and other control aspects.   The op codes are already pretty densely packed so Nvidia added a new type of op which is a pure control code.  On Kepler there is 1 control instruction for every 7 operational instructions.  Maxwell added additional control capabilities and so has 1 control for every 3 instructions.  So here's some sample sass from cuobjdump:

```assembly
                                        /* 0x001f9400fe2007e6 */
  /*0008*/    MOV R1, c[0x0][0x20];     /* 0x4c98078000870001 */
  /*0010*/    MOV R0, c[0x0][0x150];    /* 0x4c98078005470000 */
  /*0018*/    IADD R1, R1, 0x38.NEG;    /* 0x3811000003870101 */
                                        /* 0x001f9000fe2007f1 */
  /*0028*/    IADD R7, R0, -0x1;        /* 0x3910007ffff70007 */
  /*0030*/    MOV R14, RZ;              /* 0x5c9807800ff7000e */
  /*0038*/    MOV R8, RZ;               /* 0x5c9807800ff70008 */
```

On the left we have the instruction address.  Each instruction is 8 bytes so you'll see they increment by 8 in hexadecimal.  The addresses are relative to the start of the kernel.  After the address you have the instruction in a human readable format.  And the part we're interested is the last one which is the 64 bit binary representation of that instruction (printed as hexidecimal).  So you can see here that there are two 64 bit codes that have no associated instruction.  These are the control codes.  Control codes come first so each one applies to the following 3 instructions.  In MaxasGrammar.pm you'll find a function called processSassCtrlLine and it has code that looks like this:

```perl
    push @ctrl,  ($code & 0x000000000001ffff) >> 0;
    push @ctrl,  ($code & 0x0000003fffe00000) >> 21;
    push @ctrl,  ($code & 0x07fffc0000000000) >> 42;

    push @reuse, ($code & 0x00000000001e0000) >> 17;
    push @reuse, ($code & 0x000003c000000000) >> 38;
    push @reuse, ($code & 0x7800000000000000) >> 59;
```

This shows the breakdown of how the control code is split into 3 sections, one for each of the associated instructions.  I also split this up further into control and reuse codes.  The reuse codes map to the reuse flags that you'll find in the sass like this:

```assembly
XMAD R21, R17.H1, R2.reuse, R12;
XMAD R19.CC, R20.H1.reuse, R2.reuse, R5;
XMAD.CLO R4, R20.reuse, R2.reuse, RZ;
```

So these already have a manifestation in the sass (and I explain in more detail in the [sgemm](https://github.com/NervanaSystems/maxas/wiki/SGEMM#calculating-c-register-banks-and-reuse) document).  The control codes do not have any readable manifestation in the sass besides the hex value and are meant to be hidden from you.  So I split them out and output them in a format that you can see and manipulate.  So here is a sample of that format taken from the start of the sgemm code:

```assembly
--:-:1:-:1      S2R tid, SR_TID.X;   // Set Dep 1
--:-:2:-:1      S2R bx,  SR_CTAID.X; // Set Dep 2
--:-:3:-:1      S2R by,  SR_CTAID.Y; // Set Dep 3
01:-:-:-:1      ISETP.GE.AND P0, PT, tid, 128, PT; // Wait Dep 1
--:-:-:-:1      LOP.AND tid31, tid, 31;
--:-:-:-:1      BFE.U32 tid4, tid, 0x205; // 2 bits at position 5
--:-:-:-:1      MOV k, c[0x0][0x14c];
--:-:-:-:1      BFE.U32 tid7, tid, 0x301; // 3 bits at position 1
--:-:-:-:1      LOP.AND tid128, tid, 128;
--:-:-:-:1      LOP.AND readBs, tid, 0x70;
--:-:-:-:1      SHL tid31_4, tid31, 4;
--:-:-:-:1      LOP.AND tid1, tid, 1;
--:-:-:-:1      IADD k, k, -8;
--:-:-:-:1      LOP.AND zOffset, tid, -32;
--:-:-:-:1      SHR.U32 readAs, tid128, 4;
--:-:-:-:1      LOP.AND tid96, tid, 96;
--:-:-:-:1      SHR.U32 readBs, readBs, 3;
--:-:-:-:0 @!P0 MOV ldx4, c[0x0][0x150];
--:-:-:-:1      STS.128 [zOffset + 4x<16*128>], RZ;
--:-:-:-:1  @P0 MOV ldx4, c[0x0][0x154];
--:-:-:-:1      ISCADD writeS, tid4, tid31_4, 9;
06:-:-:-:1      SEL blk, by, bx, P0;               // Wait Dep 2 & 3
```

In MaxasGrammar.pm you'll find a function named printCtrl that looks like this:

```perl
sub printCtrl
{
    my $code = shift;

    my $stall = ($code & 0x0000f) >> 0;
    my $yield = ($code & 0x00010) >> 4;
    my $wrtdb = ($code & 0x000e0) >> 5;  # write dependency barier
    my $readb = ($code & 0x00700) >> 8;  # read  dependency barier
    my $watdb = ($code & 0x1f800) >> 11; # wait on dependency barier

    $yield = $yield ? '-' : 'Y';
    $wrtdb = $wrtdb == 7 ? '-' : $wrtdb + 1;
    $readb = $readb == 7 ? '-' : $readb + 1;
    $watdb = $watdb ? sprintf('%02x', $watdb) : '--';

    return sprintf '%s:%s:%s:%s:%x', $watdb, $readb, $wrtdb, $yield, $stall;
}
```

This pretty clearly shows how the 17 bits are broken down into 5 sections.  I also do a little remapping of the codes to make them easier to read.  A dash is basically a placeholder and is the absence of any control information in that section.  I used dashes because the control codes, when present, stand out better against dashes than zeros.  So here's a more detailed breakdown of each section and I'll work my way from right to left:

## Stall Counts

The main purpose of the stall count is to deal with pipeline latency and throughput.  For many of the instructions the pipeline depth is 6 clocks.  If you don't have 5 other instructions to execute in between using the results from an instruction then stalls will need to be added to make up the difference.  Also, some instructions operate at less than full throughput and gaps need to be added in between instructions of the same type to enforce that throughput (though if you forget, the hardware seems to do this for you automatically).

So here the all the values that are legal for the stall count:

```
    f : 15 clocks*
    e : 14 clocks*
    d : 13 clocks*
    c : 12 clocks*
    b : 11 clocks
    a : 10 clocks
    9 :  9 clocks
    8 :  8 clocks
    7 :  7 clocks
    6 :  6 clocks
    5 :  5 clocks
    4 :  4 clocks
    3 :  3 clocks
    2 :  2 clocks
    1 :  1 clocks  
    0 :  0 clocks, dual issue


```

Additional notes:

  * Stalling 0 means to dual issue that instruction with the following instruction.  This is only legal for instructions using two different chip resources (like a cuda core and a memory unit).
  * For stalls 12-15 the yield hint is required to be set in addition to the stall count.  Not setting this additional flag will give you fewer stall clocks.  I'm not sure what these stall counts are supposed to mean when this flag is not set.  But it seems to not be important.  
  * The most useful of the higher stall counts is the 13 count which is how long you need to wait to immediately use a predicate that is set by any instruction.
  * Some instructions with no pipeline require a minimum stall count to operate correctly.  Examples of this include BAR, BRA, CAL, RET, and EXIT which require a stall count of at least 5.
  * Memory operations have delayed access to their operands so it's possible to update their operands just prior to executing them without needing to stall the full pipeline depth.  Global memory has a longer operand read latency at 4 clocks but shared is quicker at 2 clocks.
  * By using maxas SCHEDULE\_BLOCKs you can largely ignore setting stall counts as this is handled for you, as well as optimally ordering instructions for the highest throughput.

## Yield Hint Flag

The yield hint flag is a single bit.  Instead of a '1' I use a 'Y' to set this flag.  The scheduler for the most part manages your warp yields for you but there are times when giving it a hint to yield can speed things up.  In my sgemm implementations I add a yield flag in the middle of each block of 64 FFMAs.  This seems to balance the work better and not let any one warp run too far ahead and get stuck at a barrier so that it can no longer participate in TLP.  If you have code with high levels of ILP you might want to experiment with limited use of this flag to see if your performance is increased.

Within a block or even across blocks of a kernel, yielding has no clock cost, provided work is ready to go in another warp.  There are some odd exceptions to this for highly dependent code where every instruction needs to be stalled the full pipeline depth of 6 (ILP=1).  This can give you the impression that yields do have a cost but this is not generally the case.  In these cases the cost is dependent on the number of active warps present so if you need to have code with ILP=1 you'll want to have 6 or more active warps per scheduler (to cover the stalls) and have any additional warps be a multiple of 3 (for some mysterious reason of the scheduler).

The yield hint is also required in conjunction with the higher stall counts (12-15) as described above.

## Write Dependency Barriers

There are many types of instructions that output results but don't operate at a fixed pipeline latency.  Variable latency instructions fall into two categories: memory operations, and shared resource operations like double precision ops, special function unit ops, or other low throughput ops.  Since the latency is variable we can't reliably use a stall count to wait for the results.  Instead we set and wait on a barrier.   Maxwell has 6 barrier resources for this purpose.  The raw control codes use values 0-5 but I remap these to 1-6.  I find these values easier to read for reasons that will become apparent.  So for any instruction that writes to a register and operates at variable latency, you'll want to set one of these barriers, or at least put this instruction behind another that does (though you have to be careful with this).

It is illegal to set a write dependency barrier on an instruction that doesn't write out to a register (like a memory store).

Barriers take one additional clock cycle to become active on top of the clock consumed by the instruction doing the setting.  So you may need to also use a stall count of 2 if you intend to wait with the subsequent instruction.

## Read Dependency Barriers

This type of barrier is used for instructions that don't use an operand collector and hence don't buffer their operands (mainly memory ops).  These instructions also operate at variable latency and hence to avoid write after read hazards we need to set and wait on a barrier before we can overwrite the operand values.  The barrier resources used for this are shared with the write dependency barriers, which I also number 1-6.

This also takes one additional clock to become active.

Read barriers typically have a latency of about 20 clocks (the time it takes to set the memory instruction in flight).  The complete instruction itself can take many more clocks than this.

## Wait Dependency Barrier Flags

Here is how you wait on a previously set barrier of either type.  You can wait on more than one barrier at at time so instead of working off the barrier number directly, a bit mask is used.  Here are the mask values (in hex) and corresponding barrier numbers:

```
    01 : 1
    02 : 2
    04 : 3
    08 : 4
    10 : 5
    20 : 6
```

So you can see for the first two barriers the wait mask and barrier number are the same.  This makes the code easier to read for these most frequently used barriers.  To wait on, say, both barriers 1 and 2 you'd use 03 as the mask.

Waiting on a barrier that isn't set or is already signaled seems to not add any additional cost.

## Predicated Execution and Dependency Barriers

When you are mixing predicates and barrier flags the instructions behaves as if there was no predicate at all.  So you can not count on predicated sets or waits of the barriers.  If you need this kind of functionality the instructions need to be part of a branch.  If a predicate is warp uniform and used on a memory load, for example, then the wait time will be substantially less if there ends up being nothing to load.

# Control Notation Examples

S2R is a variable latency (about 20 clocks) load like instruction.  Here we use one barrier per S2R.  I think something like this could all share a single barrier, but since we have 6 available we dedicate one to each.  I need to further investigate the limits of using a single barrier for multiple instructions.

**Update:** It is not wise to share a barrier between S2R instructions.  These can be processed out of order.  Use an explicit barrier for each.  This is also what ptxas does.

Note the pipeline depth of ISETP is 13 clocks so we need to stall 13 prior to reading P0 in the SEL instruction (in addition to waiting on barriers 2 and 3).  Also note that we're using the required yield flag to get the 13 count to work correctly.  Without this flag, the actual stall would be just 6 clocks and the results would be non-deterministic.

```assembly
--:-:1:-:1      S2R tid, SR_TID.X;   // Set Dep 1
--:-:2:-:1      S2R bx,  SR_CTAID.X; // Set Dep 2
--:-:3:-:1      S2R by,  SR_CTAID.Y; // Set Dep 3
01:-:-:Y:d      ISETP.GE.AND P0, PT, tid, 128, PT; // Wait Dep 1   Stall 13
06:-:-:-:1      SEL blk, by, bx, P0;               // Wait Dep 2,3
```


---


Here we are dual issuing FFMA's with the shared load instructions.  We're also relying on the memory ops to be performed in source order so that they can share a barrier.  This seems to be pretty reliable for a reasonable number of memory ops.  Note the stall count of 2 on the final LDS since we're waiting on the set barrier in the following instruction.  We don't want to modify the LDS operands until the instruction signals to the barrier that it is done with the operand.  This takes about 20 clocks which is much less time than it takes to complete the load.  Finally we wait on barrier 1 before using the results of the LDS.

```assembly
--:-:-:-:0      FFMA cx02y00, j0Ax02, j0By00, cx02y00;
--:-:-:-:1      LDS.U.128 j1Ax00, [readAs + 4x<1*128 + 00>];
--:-:-:-:1      FFMA cx02y01, j0Ax02, j0By01, cx02y01;
--:-:-:-:0      FFMA cx00y01, j0Ax00, j0By01, cx00y01;
--:-:-:-:1      LDS.U.128 j1By00, [readBs + 4x<1*128 + 00>];
--:-:-:-:1      FFMA cx00y00, j0Ax00, j0By00, cx00y00;
--:-:-:-:0      FFMA cx03y00, j0Ax03, j0By00, cx03y00;
--:-:-:-:1      LDS.U.128 j1Ax64, [readAs + 4x<1*128 + 64>];
--:-:-:-:1      FFMA cx03y01, j0Ax03, j0By01, cx03y01;
--:-:-:-:0      FFMA cx01y01, j0Ax01, j0By01, cx01y01;
--:2:1:-:2      LDS.U.128 j1By64, [readBs + 4x<1*128 + 64>]; // Set Dep 1,2  Stall 2 

02:-:-:-:1      IADD readAs, readAs, 16; // Wait Dep 2
--:-:-:-:1      IADD readBs, readBs, 16;

01:-:-:-:1      FFMA cx02y00, j1Ax02, j1By00, cx02y00; // Wait Dep 1
```


---


Here we load two textures setting two barriers.  But you can see just prior to a texture load we update its operand.  Notice that we don't need to stall a full count of 6 prior to using it since the texture doesn't start reading that register for at least 4 clocks.  After the wait is complete and the STS executes, it is now safe to increment the track0 variable.  Note that we didn't need to use a read barrier in this case since the write barrier already signaled that the instruction was fully completed.  Making double use of barriers is a good way to cut out some of the latencies in your code, increasing the overall chances for TLP.

Note the IADDs can be dual issued with the STS instructions, as well as the BAR.SYNC.  Note the BAR.SYNC needs a 5 clock stall.  And note that in order to safely XOR writeS, we don't need to set a read barrier on the STS instructions.  This is because the BAR.SYNC ensures that all memory stores are complete before returning.  BAR.SYNC does _not_ guarantee that any memory loads are complete so you'll still have to use barriers for those.  It is legal to add wait flags to a BAR.SYNC (or any instruction for that matter I think).  And as I said, grouping barriers together can have some benefits.

```assembly
--:-:-:-:2      XMAD.PSL.CBCC track0, ldx.H1, xmad_t0.H1, track0; // Stall 6 - 4 = 2
--:-:1:-:1      TLD.B.LZ.P loadX0, track0, tex, 0x0, 1D, 0xf; // Set Dep 1
--:-:2:-:1      TLD.B.LZ.P loadX4, track4, tex, 0x0, 1D, 0xf; // Set Dep 2

01:-:-:-:1      STS.128 [writeS + 4x<0*128>], loadX0; // Wait Dep 1
--:-:-:-:0      IADD track0, track0, ldx8;
02:-:-:-:1      STS.128 [writeS + 4x<4*128>], loadX4; // Wait Dep 2
--:-:-:-:0      IADD track4, track4, ldx8;
--:-:-:-:5      BAR.SYNC 0;

--:-:-:-:1      LOP.XOR writeS, writeS, 4x<16*128>;
```


---

Ok, here is a tricky example from the sgemm code that produces an error and that I don't yet fully understand.  In this example the memory stored at zOffset has been previously set to zero.  We use LDS.U.128 to initialize 64 registers to zero instead of using 64 MOV ops to save some clocks and keep the kernel smaller.

Note that we have 20 LDS commands all sharing the same barrier here.  The error that occurs here is of the write after read class.   You'll notice that zOffset and j1Ax00 share the same register: 80.  What is happening is that LDS.U.128 j1Ax00 is completing and hence zOffset is being modified before some of the LDS.U.128 czXX loads are able to enter flight.  So the values that czXX get initialized to are from a different area of shared memory than what we intended (and don't get set to zero).  This is happening despite the fact that there is a barrier between the write after read hazard.

I think the problem here is that we are just overwhelming the memory unit with LDS.128 ops and source ordering is no longer reliable.  We can try using additional barriers but we only have 6 and this is not enough to make the problem go away.  I tried a MEMBAR.CTA but this doesn't help either.  One way that I can get this code to work is by moving the C initialization further up and behind a couple STS instructions and a BAR.SYNC.  I believe that stores and loads are order preserving and so the bar.sync then effectively waits for the loads to complete as well.  The simpler way that fixes this issue is to just not share register 80 for the two variables.

So I think the lesson here is that for extremely memory op dense code, you can't rely on barriers to save you from hazards.  You'll need to make sure enough clocks transpire to dilute the density or you need to mix the loads with some stores which you can synchronize against.  And indeed inserting a bunch of NOPs with high stall counts does fix the problem as well.  This is probably not the kind of error you'd encounter in normal code, but only when trying to be overly clever with memory ops.  But at least you know this kind of reordering is possible.

**Update:**  This problem seems to have been corrected on GM204 and GM200 as I have not been able to duplicate it since.  The instruction queue depth for shared memory instructions appears to be very large on this hardware (like on the order of 50).  Global memory instructions have a much smaller queue (like on the order of 7-8) but execution is properly blocked when this queue fills up for a warp.  Warps seem to maintain their own queues and are not shared.

```assembly
<REGISTER_MAPPING>

    // Reuse a register from the second blocking registers
    80      : zOffset

    // Double buffered register blocking used in vector loads.
    64-79   : j0Ax<00-03|64-67>, j0By<00-03|64-67>
    80-95   : j1Ax<00-03|64-67>, j1By<00-03|64-67>

</REGISTER_MAPPING>

// Initialize C registeres to zero
--:-:-:-:1      LDS.U.128 cz00, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz04, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz08, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz12, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz16, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz20, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz24, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz28, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz32, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz36, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz40, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz44, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz48, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz52, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz56, [zOffset + 4x<16*128>];
--:-:-:-:1      LDS.U.128 cz60, [zOffset + 4x<16*128>];

// Preload the fist lines of A and B from shared
--:-:-:-:1      LDS.U.128 j0Ax00, [readAs + 4x<0*128 + 00>];
--:-:-:-:1      LDS.U.128 j0By00, [readBs + 4x<0*128 + 00>];
--:-:-:-:1      LDS.U.128 j0Ax64, [readAs + 4x<0*128 + 64>];
--:-:1:-:1      LDS.U.128 j0By64, [readBs + 4x<0*128 + 64>]; // Set Dep 1

LOOP:
--:-:-:-:1      ISETP.LE.AND P0, PT, track0, end, PT;

// Start computing with preloaded data and load the next line
01:-:-:-:0      FFMA cx02y00, j0Ax02, j0By00, cx02y00; // Wait Dep 1
--:-:-:-:1      LDS.U.128 j1Ax00, [readAs + 4x<1*128 + 00>];
```

-- Scott