# stereopsis-ros

Another aspect of project stereopsis, drivers and such.

## Notes to for setup
- Make sure you have c compiler if not
~~~
apt install build-essential
~~~
- Clone [pico-flexx-driver repo](https://github.com/code-iai/pico_flexx_driver)  to the src/ directory as another package. Follow the instructions on the aforementioned repository, it would ask you to get the royale sdk.

> **Qt5 error:** During one of the attempts, qt5 caused a problem it was solved by
~~~
apt-get install qt5-default
~~~
Although I'm not sure whether it is necessary since the error was thrown by a sample code provided by royale, haven't tried whether it would be fatal to omit the specific sample.

After this you should be able to run the command to grab images from the pico flexx
~~~
roslaunch pico_flexx_driver pico_flexx_driver.launch
~~~
