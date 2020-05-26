# Reduce
run: reduce
	./reduce 1073741824

reduce: reduce.cc 
	g++ -std=c++11 -O3 -g -pthread  $< -o $@ -lboost_system  -lboost_thread

clean:
	rm *.out *.err *.log reduce

# Condor
submit: reduce
	condor_submit hw1.cmd

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm
