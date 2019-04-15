# DO NOT EDIT
# This makefile makes sure all linkable targets are
# up-to-date with anything they link to
default:
	echo "Do not invoke directly"

# Rules to remove targets that are older than anything to which they
# link.  This forces Xcode to relink the targets from scratch.  It
# does not seem to check these dependencies itself.
PostBuild.mpc.Debug:
/Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/Debug/mpc:
	/bin/rm -f /Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/Debug/mpc


PostBuild.mpc.Release:
/Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/Release/mpc:
	/bin/rm -f /Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/Release/mpc


PostBuild.mpc.MinSizeRel:
/Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/MinSizeRel/mpc:
	/bin/rm -f /Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/MinSizeRel/mpc


PostBuild.mpc.RelWithDebInfo:
/Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/RelWithDebInfo/mpc:
	/bin/rm -f /Users/mac/Developer/udacity/CarND/SDCE_nanodegree/Term2/MPC_Quizzes/mpc_to_line/build-xcode/RelWithDebInfo/mpc




# For each target create a dummy ruleso the target does not have to exist
