import os
env = Environment()
env['ENV']['TERM'] = os.environ['TERM']
env['CXX'] = "clang++"
env['CXXFLAGS'] = "-std=c++11 -O3"

def CheckPKGConfig(context, version):
     context.Message( 'Checking for pkg-config... ' )
     ret = context.TryAction('pkg-config --atleast-pkgconfig-version=%s' % version)[0]
     context.Result( ret )
     return ret

def CheckPKG(context, name):
     context.Message( 'Checking for %s... ' % name )
     ret = context.TryAction('pkg-config --exists \'%s\'' % name)[0]
     context.Result( ret )
     return ret

# Configuration:
conf = Configure(env, custom_tests = { 'CheckPKGConfig' : CheckPKGConfig,
                                       'CheckPKG' : CheckPKG })

if not conf.CheckPKGConfig('0.15.0'):
     print 'pkg-config >= 0.15.0 not found.'
     Exit(1)

if not conf.CheckPKG('opencv >= 2'):
     print 'opencv 2 not found.'
     Exit(1)

# Your extra checks here

env = conf.Finish()

# Now, build:

env.ParseConfig('pkg-config --cflags --libs opencv')

files = Glob('*.cpp') #["Classify.cpp", "feature.cpp", "main.cpp"]

env.Program(target="obj", source=files)
