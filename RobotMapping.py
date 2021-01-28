
# Robot Mapping System 
 
import rospy
import numpy as np
import unittest
import time as t
from collections import defaultdict
from scipy.linalg import block_diag

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from cylinder.msg import cylDataArray
from cylinder.msg import cylMsg
from sensor_msgs.msg import Imu
from matplotlib import pyplot as plt 

from Relative2AbsolutePose import Relative2AbsolutePose
from Relative2AbsoluteXY import Relative2AbsoluteXY
from Absolute2RelativeXY import Absolute2RelativeXY
from pi2pi import pi2pi
from mapping import mapping
# landmarks' most recent absolute coordinate
landmark_abs_ = defaultdict(list)
seenLandmarks_ = []
# State Transition Model
F_ = []
# Control-Input Model
W_ = []
# dimension of robot pose
dimR_ = 3

# initial array for print data to text file
robotx = []
roboty = []
robotth = []
lab = []
dvv = []
solution = []

class Robot():
    def __init__(self, pose, pos_Cov, sense_Type):

        self.x = pose[0][0]
        self.y = pose[1][0]
        self.theta = pose[2][0]
        self.poseCovariance = pos_Cov
        self.senseType = sense_Type

    def setPose(self, new_pose):
        # print 'setpose', self
        self.x = new_pose[0][0]
        self.y = new_pose[1][0]
        self.theta = new_pose[2][0]

    def getPose(self):
        return [[self.x], [self.y], [self.theta]]

    def setCovariance(self, new_Cov):

        self.poseCovariance = new_Cov

    def getCovariance(self):

        return self.poseCovariance

    def move(self, robotCurrentAbs, u):

        [nextRobotAbs, H1, H2] = Relative2AbsolutePose(robotCurrentAbs, u)

        self.x = nextRobotAbs[0][0]
        self.y = nextRobotAbs[1][0]
        self.theta = nextRobotAbs[2][0]
        return nextRobotAbs, H1, H2

    def sense(self, robotCurrentAbs, landmarkAbs):

        if self.senseType == 'Vision':
            [measurement, H1, H2] = Absolute2RelativeXY(robotCurrentAbs, landmarkAbs)

        else:
            raise ValueError('Unknown Measurement Type')

        return measurement, H1, H2

    def inverseSense(self, robotCurrentAbs, measurement):

        if self.senseType == 'Vision':
            [landmarkAbs, H1, H2] = Relative2AbsoluteXY(robotCurrentAbs, measurement)

        else:
            raise ValueError('Unknown Measurement Type')

        return landmarkAbs, H1, H2


class LandmarkMeasurement():
    def __init__(self, meas_Cov):
        self.measurementCovariance = meas_Cov

    def setCovariance(self, new_Cov):
        self.measurementCovariance = new_Cov

    def getCovariance(self):
        return self.measurementCovariance


class Motion():
    def __init__(self, motion_command, motion_Cov):
        self.u = motion_command
        self.motionCovariance = motion_Cov

    def setCovariance(self, new_Cov):
        self.motionCovariance = new_Cov

    def getCovariance(self):
        return self.motionCovariance

    def setMotionCommand(self, motionCommand):
        self.u = motionCommand

    def getMotionCommand(self):
        return self.u


class KalmanFilter(Robot):
    def __init__(self, mean, covariance, robot):

        self.stateMean = mean
        self.stateCovariance = covariance
        self.robot = robot

    def setStateMean(self, mean):

        self.stateMean = mean

    def getStateMean(self):

        return self.stateMean

    def setStateCovariance(self, covariance):

        self.stateCovariance = covariance

    def getStateCovariance(self):

        return self.stateCovariance

    def predict(self, motion, motionCovariance):
        
        global robotx, roboty, robotth

        
        #   get robot current pose
        RobotCurrentPose = self.robot.getPose()
        #   move robot given current pose and u
        [nextRobotAbsPose, F, W] = self.robot.move(RobotCurrentPose, motion)
        #   predict state mean
        stateMean = self.getStateMean()
        #   predict state covariance
        stateCovariance = self.robot.getCovariance()
        new_Cov = np.dot(np.dot(F, stateCovariance), np.transpose(F)) + np.dot(np.dot(W, motionCovariance), np.transpose(W))
        #   set robot new pose
        self.robot.setPose(nextRobotAbsPose)
        #   set robot new covariance
        self.robot.setCovariance(new_Cov)
        #   set KF priorStateMean
        priorStateMean = self.getStateMean()
        priorStateMean[0:3] = nextRobotAbsPose
        self.setStateMean(priorStateMean)
        #   set KF priorStateCovariance
        priorStateCovariance = self.getStateCovariance()
        priorStateCovariance[0:3,0:3] = np.transpose(new_Cov)
        self.setStateCovariance(priorStateCovariance)
        robotx.append(nextRobotAbsPose[0][0])
        roboty.append(nextRobotAbsPose[1][0])                  
        robotth.append(nextRobotAbsPose[2][0])                

        return priorStateMean, priorStateCovariance

    def update(self, measurement, measurementCovariance, new):
        global seenLandmarks_
        global dimR_
        global lab
        global solution
        #   get robot current pose
        robotCurrentAbs = self.robot.getPose()
        # get landmark absolute position estimate given current pose and measurement (robot.sense)
        [landmarkAbs, G1, G2] = self.robot.inverseSense(robotCurrentAbs, measurement)
        #   get KF state mean and covariance
        stateMean = self.getStateMean()
        stateCovariance = self.getStateCovariance()
        # if new landmark augment stateMean and stateCovariance
        if new:
            # print 'new'
            stateMean = np.concatenate((stateMean, [[landmarkAbs[0]], [landmarkAbs[1]]]), axis=0)
            Prr = self.robot.getCovariance()
            # print stateMean
            if len(seenLandmarks_) == 1:
                Plx = np.dot(G1, Prr)
            else:

                lastStateCovariance = self.getStateCovariance()
                Prm = lastStateCovariance[0:3, 3:]
                Plx = np.dot(G1, np.bmat([[Prr, Prm]]))
            Pll = np.array(np.dot(np.dot(G1, Prr), np.transpose(G1))) + np.array(
                np.dot(np.dot(G2, measurementCovariance), np.transpose(G2)))
            P = np.bmat([[stateCovariance, np.transpose(Plx)], [Plx, Pll]])
            stateCovariance = P
            # new cylinder detected and the dimension of statemean and statecovariance changes, need to update
            self.setStateMean(self,stateMean)
            self.setStateCovariance(self,stateCovariance)
        else:
        # if old landmark stateMean & stateCovariance remain the same (will be changed in the update phase by the kalman gain)
            # calculate expected measurement
            # get the index of the cylinder observed to get its previous position and calculate exected measurement
            label = measurement[2]
            vec1 = mapping(seenLandmarks_.index(label) + 1)
            landmarkPriorAbs = [[stateMean[dimR_ + vec1[0] - 1][0]], [stateMean[dimR_ + vec1[1] - 1][0]]]
            [expectmeasurement, Hr, Hl] = self.robot.sense(robotCurrentAbs, landmarkPriorAbs)
            # get measurement
            Z = [[measurement[0]], [measurement[1]]]
            # Update
            x = stateMean
            # y = Z - expectedMeasurement
            y = np.array(Z) - np.array(expectmeasurement)
            # H = [Hr, 0, ..., 0, Hl] position of Hl depends on when was the landmark seen?
            H = np.reshape(Hr, (2, 3))
            for i in range(0, seenLandmarks_.index(label)):
                H = np.bmat([[H, np.zeros([2, 2])]])
            H = np.bmat([[H, np.reshape(Hl, (2, 2))]])
            for i in range(0, len(seenLandmarks_) - seenLandmarks_.index(label) - 1):
                H = np.bmat([[H, np.zeros([2, 2])]])
            # compute S            
            S = np.dot(np.dot(H, stateCovariance), np.transpose(H)) + measurementCovariance

            if (abs(S) < 0.000001).all():
                print('Non-invertible S Matrix')
                raise ValueError
                return
            else:
                #   calculate Kalman gain
                K = np.dot(np.dot(stateCovariance, np.transpose(H)), np.linalg.inv(S))
                #   compute posterior mean
                # simple filtering approach, if the correctness value is too big, consider it as an error, don't update
                E = np.array(np.dot(K,y))
                if abs(E[0])>0.1:
                    posteriorStateMean = np.array(x)
                else:
                    posteriorStateMean = np.array(x)+ np.array(np.dot(K, y))


                # compute posterior covariance
                I = np.identity(len(np.dot(K,H)))
                posteriorStateCovariance = np.dot(I - (np.dot(K, H)),stateCovariance)
                # check theta robot is a valid theta in the range [-pi, pi]
                posteriorStateMean[2][0] = pi2pi(posteriorStateMean[2][0])
                # update robot pose
                robotPose = posteriorStateMean[0:3]
                lab.append(str(label))
                # set robot pose
                self.robot.setPose(robotPose)
                # updated robot covariance
                robotCovariance = posteriorStateCovariance[0:3, 0:3]
                # set robot covariance
                self.robot.setCovariance(robotCovariance)
                # set posterior state mean
                KalmanFilter.setStateMean(self, posteriorStateMean)
                # set posterior state covariance
                KalmanFilter.setStateCovariance(self, posteriorStateCovariance)
                # print 'robot absolute pose : ', robotAbs
                vec = mapping(seenLandmarks_.index(label) + 1)
                landmark_abs_[int(label) - 1] = [[[stateMean[dimR_ + vec[0] - 1][0]], [stateMean[dimR_ + vec[1] - 1][0]]]]
            for i in range(0, len(landmark_abs_)):
                print 'landmark absolute position : ', i + 1, ',', np.median(landmark_abs_[i], 0)

            solution = landmark_abs_
            return posteriorStateMean, posteriorStateCovariance


class SLAM(LandmarkMeasurement, Motion, KalmanFilter):
    def callbackOdometryMotion(self, msg):
        #   read msg received
        x = msg.twist.twist.linear.x
        theta = msg.twist.twist.angular.zm
        #   You can choose to only rely on odometry or read a second sensor measurement
        #   compute dt = duration of the sensor measurement
        current_time = msg.header.stamp.to_sec()
        dt = (current_time - self.last_time)
        self.last_time = current_time
        #   compute command
        dx = x * dt
        # dtheta = self.dv * dt
        dtheta = self.dtheta
        dthodom = theta*dt
        # dtheta = theta * dt
	if dx<0.01:
            u = np.array([[dx], [0], [dthodom]])
	else:
	    u = np.array([[dx], [0], [dtheta]])
        # set motion command
        self.motion.setMotionCommand(u)
        # get covariance from msg received
        covariance = msg.twist.covariance
        covIMU = self.covIMU[8]
        # set the covariance of IMU to the angular covariance
        self.motion.setCovariance([[covariance[0], 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, covIMU]])
        poseCovariance = self.robot.getCovariance()
        # call KF to execute a prediction
        self.KF.predict(self.motion.getMotionCommand(), self.motion.getCovariance())

    def callbackImu(self,msg):
        global dvv
        dv = msg.angular_velocity.z
        dvv.append(dv)
        # dynamic compensation algorithm, eliminate floating data and compensate the useful data
        if dv < 0.1:
            dv = dv * 0.1
        else:
            c = (0.6-dv)/0.6    
            dv = dv*(1+c)

        current_time = msg.header.stamp.to_sec()
        dt = (current_time - self.last_time_imu)
        self.last_time_imu = current_time
        self.dtheta = dv * dt
        self.covIMU = msg.angular_velocity_covariance


    def callbackLandmarkMeasurement(self, data):
        global seenLandmarks_
        for i in range(0, len(data.cylinders)):
            # read data received
            # aligning landmark measurement frame with robot frame
            dx = data.cylinders[i].Zrobot
            dy = -data.cylinders[i].Xrobot
            label = data.cylinders[i].label
            # determine if landmark is seen for first time
            # or it's a measurement of a previously seen landamrk
            new = 0
            # if seenLandmarks_ is empty
            if not seenLandmarks_:
                new = 1
                seenLandmarks_.append(label)
            # if landmark was seen previously
            elif label not in seenLandmarks_:
                new = 1
                seenLandmarks_.append(label)
            measurement = []
            measurement.append(dx)
            measurement.append(dy)
            measurement.append(label)
            # get covariance from data received
            covariance = data.cylinders[i].covariance
            self.landmarkMeasurement.setCovariance([[covariance[0], 0.0], [0.0, covariance[3]]])
            measurementLandmarkCovariance = self.landmarkMeasurement.getCovariance()
            # call KF to execute an update
            try:
                self.KF.update(measurement, measurementLandmarkCovariance, new)
            except ValueError:
                return

    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self):
        # Initialise robot
        # get the odometer time and IMU time as the initial time
        msg = rospy.wait_for_message('/odom', Odometry)
        self.last_time = msg.header.stamp.to_sec()
        msgImu = rospy.wait_for_message('/mobile_base/sensors/imu_data',Imu)
        self.last_time_imu = msg.header.stamp.to_sec()

        #   initialise a robot pose and covariance
        robot_pose = [[0], [0], [0]]
        robot_covariance = [[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]]
        sense_Type = 'Vision'
        self.robot = Robot(robot_pose, robot_covariance, sense_Type)
        # Initialise motion
        # initialise a motion command and covariance
        motion_command = [[0], [0], [0]]
        motion_covariance = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]
        self.motion = Motion(motion_command, motion_covariance)
        # Initialise landmark measurement
        # initialise a measurement covariance
        measurement_covariance = [[0.01, 0], [0, 0.01]]
        self.landmarkMeasurement = LandmarkMeasurement(measurement_covariance)
        # Initialise kalman filter
        # initialise a state mean and covariance
        state_mean = [[0], [0], [0]]
        state_covariance = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        # initial state contains initial robot pose and covariance
        self.KF = KalmanFilter(state_mean, state_covariance, self.robot)
        # Subscribe to different topics and assign their respective callback functions
        self.dtheta = 0
        self.dv = 0
        self.covIMU = [0,0,0,0,0,0,0,0,0]
        rospy.Subscriber('/mobile_base/sensors/imu_data',Imu, self.callbackImu)
        rospy.Subscriber('/odom', Odometry, self.callbackOdometryMotion)
        rospy.Subscriber('/cylinderTopic', cylDataArray, self.callbackLandmarkMeasurement)
        
    # plot the trajectory of the robot in realtime
        plt.figure(1)
        plt.ion()
        plt.axes(xlim=(-4,4), ylim=(-1,6))
        while not rospy.is_shutdown():
            rx = self.KF.getStateMean()
            plt.plot(rx[0][0],rx[1][0],'bo')
            plt.draw()
            plt.pause(0.01)
     
        rospy.spin()
    


if __name__ == '__main__':
    print('Landmark SLAM Started...')
    # Initialize the node and name it.
    rospy.init_node('listener')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        # write the robot pose and landmark position to different files
        f = open("RobotPose.txt", "w")
        g = open("solutionFile.txt", "w")
        slam = SLAM()
        f.write('x='+str(robotx)+'\n'+'y='+str(roboty)+'\n'+'th='+str(robotth)+'\n'+'dv='+str(dvv))
        g.write(str(solution))
        g.close()
        f.close()

    except rospy.ROSInterruptException:
        pass

