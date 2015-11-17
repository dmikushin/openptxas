# Getting Started

The first step is to download the maxas [source code](https://github.com/NervanaSystems/maxas).

You can use any recent version of cuda for this, but the the nvdisasm binary must be from cuda 6.5.  I would install that and overwrite the version in cuda 7.x (after making a backup copy).  You can get cuda 6.5 here: nvdisasm from cuda 6.5: https://developer.nvidia.com/cuda-toolkit-archive

Next, you'll want to make sure you have Perl installed.  If on Windows I'd install the latest 64bit version from [Active State](http://www.activestate.com/activeperl/downloads).  If on Linux then you probably already have it installed, just make sure it's at least version 5.10 (the code makes heavy use of named regex capture groups introduced in 5.10).  It also requires 64bit as the op codes are that long and native support makes processing them much easier.

If you have the [cpanm](http://search.cpan.org/~miyagawa/App-cpanminus-1.7027/lib/App/cpanminus.pm) module installed you can download and install maxas in one fell swoop via:
```
sudo cpanm git://github.com/NervanaSystems/maxas.git
```
Otherwise from the checked out source directory, issue:
```
perl MakeFile.PL
make
make test  # optional
sudo make install
```

Next, you'll want to setup a text editor that supports custom syntax highlighting.  I'm not real crazy about Visual Studio as an editor for anything other than C++ or C#.  So I mainly use [TextPad](https://www.textpad.com/).  I've included a starter syntax file (sass.syn) for you to use.  I haven't yet populated it with every single permutation of instruction name, I've just been adding ones as I go.  You can probably convert that file to another editor's format if you were so inclined.

Next is to setup your shell environment to easily work with the software.  If in windows I'd recommend, if you haven't already, to install a decent console application like [ConEmu](https://code.google.com/p/conemu-maximus5/).  Working directly in cmd.exe is pure masochism.  I like to just edit my vcvarsall.bat script that I configure to start with every console invocation.  Here's what I have at the top:

```
DOSKEY maxas=maxas.pl $* 
DOSKEY ls=dir 
DOSKEY rm=del $* 
```

Or you could edit your system environment variables.  Ok, so from anywhere now you should be able run the maxas command.  Give it a try and you should see the usage like this:

```
Usage:

  List kernels and symbols:

     maxas.pl --list|-l <cubin_file>

  Test a cubin or sass file to to see if the assembler can reproduce 
  all of the contained opcodes. Also useful for extending the missing 
  grammar rules. Defaults to only showing failures without --all. 
  With the --reg flag it will show register bank conflicts not 
  hidden by reuse flags.

     maxas.pl --test|-t [--reg|-r] [--all|-a] <cubin_file | cuobjdump_sass_file>

  Extract a single kernel into an asm file from a cubin. 
  Works much like cuobjdump but outputs in a format that can be 
  re-assembled back into the cubin.

     maxas.pl --extract|-e [--kernel|-k kernel_name] <cubin_file> [asm_file]

  Preprocess the asm: expand CODE sections, perform scheduling. 
  Mainly used for debugging purposes. Include the debug flag to 
  print out detailed scheduler info.

     maxas.pl --pre|-p [--debug|-d] <asm_file> [new_asm_file]

  Insert the kernel asm back into the cubin.  Overwrite existing or 
  create new cubin. Optionally you can skip register reuse flag auto 
  insertion. This allows you to observe performance without any reuse
  or you can use it to set the flags manually in your sass.

     maxas.pl --insert|-i [--noreuse|-n] <asm_file> <cubin_file> [new_cubin_file]
```

So, we can't do much with the tool without some existing code to work with.  Let's experiment in the provided microbench directory.  There's already a basic kernel in microbench.cu we can build off of.  The Visual Studio project that's there is not configured to compile that file because I'm using the driver API and loading modules manually.  It is possible to use the cuda runtime with this tool.  See the comments at the bottom of microbench.cu for info on that.  Anyway, lets build the cubin to get started:

```
> nvcc -arch sm_50 -cubin microbench.cu
```

This produces microbench.cubin.  You should only need to compile the starter cubin once unless you end up changing kernel params or globals or shared memory.  We can now inspect the cubin with maxas:

```
> maxas.pl -l microbench.cubin

Kernel: microbench (Linkage: GLOBAL, Params: 3, Size: 256, Registers: 5, SharedMem: 4096, Barriers: 1)
```

We could next extract the sass with this command (this prints to stdout):

```
> maxas.pl -e microbench.cubin
```

But since I already have the microbench.sass file there we can just use that one.  Let's insert that kernel code into the cubin.  For this to work correctly you need to make sure the "Kernel" comment is preserved at the top of the file and that it matches the name and signature of the kernel you're trying to replace:

```
> maxas.pl -i microbench.sass microbench.cubin
```

So that cubin will now have our custom kernel code.  We can build the microbench project and run it. It should show us the typical latency of the S2R op (using threadIdx isn't as cheap as you might expect, though there are many tricks to hide that latency).

To run this kernel in the debugger, you'll need to go to the Debug menu of Visual Studio and set a new breakpoint at a function name.  Put in the name of the kernel.  Then in the Nsight menu choose Start CUDA Debugging.  This should bring you to the start of the kernel SASS.  You don't need to have the debug configuration chosen for this to work.

Ok, so that's it for the basics.  I'd move on to the [sgemm](https://github.com/NervanaSystems/maxas/wiki/SGEMM) documentation as that walks you through a fairly complicated kernel using all the features of the assembler.  Then after reading the rest of the detailed documentation, I'd start playing with your old code that you're pretty familiar with.  Export the sass and get a feel for how ptxas structures your code and uses the control notation.  Also learn all the basic sass syntax.

If you run into any trouble, join the mailing list.  I plan to pretty aggressively support this project and want to make development as painless as possible.  I'm also keen on feature ideas you may have.  Good luck..

--Scott