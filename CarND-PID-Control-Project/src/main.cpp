#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_last_of("]");
    if (found_null != std::string::npos) {
        return "";
    }
    else if (b1 != std::string::npos && b2 != std::string::npos) {
        return s.substr(b1, b2 - b1 + 1);
    }
    return "";
}

int main()
{
    uWS::Hub h;

    PID pid_steer;
    // TODO: Initialize the pid variable.

    pid_steer.Init(2e-1,7.5e-7,1.35267);

    h.onMessage([&pid_steer](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        if (length && length > 2 && data[0] == '4' && data[1] == '2')
        {
            auto s = hasData(std::string(data).substr(0, length));
            if (s != "") {
                auto j = json::parse(s);
                std::string event = j[0].get<std::string>();
                if (event == "telemetry") {
                    // j[1] is the data JSON object
                    double cte = std::stod(j[1]["cte"].get<std::string>());
                    double speed = std::stod(j[1]["speed"].get<std::string>());
                    double angle = std::stod(j[1]["steering_angle"].get<std::string>());
                    double throttle_value =  std::stod(j[1]["throttle"].get<std::string>());
                    double steer_value;//, throttle_value;
                    /*
                    * TODO: Calcuate steering value here, remember the steering value is
                    * [-1, 1].
                    * NOTE: Feel free to play around with the throttle and speed. Maybe use
                    * another PID controller to control the speed!
                    */
                    // Calculate steer values using PID model
                    pid_steer.UpdateError(cte);
                    steer_value = pid_steer.TotalError();
                    steer_value = fmin(fmax(steer_value,-1),1);

                    // Calculate throttle value based on the steer_value & current angle: a low throttle_value is correlated to a large steer_value
                    double Kd_angle =1.3*fabs(rad2deg(steer_value))/50.0 + 0.7*fabs(angle)/25;
                    throttle_value = 1 - Kd_angle;
                    if(angle >= 7){throttle_value=-1;}
                    throttle_value = fmin(fmax(throttle_value,-1),1);
                    // DEBUG
                    std::cout << "CTE: " << cte <<" speed: "<< speed << " Steering Value: " << rad2deg(steer_value) <<
                              " angle: "<<angle<<" Kd_angle: "<< Kd_angle<<" Throttle: "<<throttle_value << std::endl;
                    std::cout <<  "CTE: " << cte << " p_error: "<<pid_steer.p_error << " d_error: "<<pid_steer.d_error
                              << " i_error: " << pid_steer.i_error << std::endl;
                    json msgJson;
                    msgJson["steering_angle"] = steer_value;
//          msgJson["throttle"] = 0.5;
                    msgJson["throttle"] = throttle_value;
                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    std::cout << msg << std::endl;
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1)
        {
            res->end(s.data(), s.length());
        }
        else
        {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
