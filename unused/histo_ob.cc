#include <iostream>
#include <vector>
//#include <unistd.h>
#include <string.h>
#include "user_space.h"
#include "user_quote_query.h"
#include "user_interface.h"
#include "json_parse_interface.h"
#include <chrono> //timer class
#include <stdlib.h>
#include <algorithm>
#include <deque>


namespace foo {
	struct Cylinder {
		uint16_t n_tick;											//Number of ticks we want to record <256
		uint8_t n_level;											//Number of levels we want to record <256
		uint16_t tick_tail;											//Point to the tail of the timeseries
		uint8_t ob_depth;											//Lv 2 ob usuaully 5
		float min_tick_div;
		std::vector<int16_t> last_tick_row;							//Record the ob info of the last tick as a single vector		
		std::vector<int16_t> record;								//Record of historical ob info
		uint16_t non_zero_left;
		uint16_t non_zero_right;
	};

	struct NewOrder {
		int64_t order_id;
		float price;
		int16_t quantity;
	};

	struct Order {
		int64_t order_id
		float price;
		int8_t quantity;
		int8_t queue_size;
		int8_t queue_posit;
		int8_t sign;
	};

	/*
		Initialize the data structure with width and height
	*/
	void init_cylinder(struct Cylinder * cy, uint16_t ticks, uint8_t levels, float min_tick, uint8_t obd) {
		cy->n_tick = ticks;
		cy->n_level = levels;
		cy->min_tick_div = 1.0 / min_tick;
		// UPDATE: use 2D vector here
		cy->record = std::vector<std::vector<int16_t>>(cy->n_tick, std::vector<int16_t> (cy->n_level, 0));
		
		cy->last_tick_row = std::vector<int16_t>(levels, 0);
		cy->non_zero_left = 0;		// inclusive
		cy->non_zero_right = 0;		// inclusive

		cy->tick_tail = ticks - 1;
		cy->ob_depth = obd; //1 for level data
	}

	/*
		Reset internal matrix. UPDATE: use this method only at the end of trading sessions
	*/
	void reset_cylinder_record(struct Cylinder * cy) {
		cy->record = std::vector<std::vector<int16_t>>(cy->n_tick, std::vector<int16_t>(cy->n_level, 0));
		cy->last_tick_row = std::vector<int16_t>(cy->n_level, 0);
		cy->non_zero_left = 0;		//inclusive
		cy->non_zero_right = 0;		// exclusive
	}

	/*
		UPDATE: Use *tick as a param instead
	*/
	void update_cylinder(struct Cylinder * cy, const Tick *tick) {
		// First update last_tick_row and its right/left bound position
		if (cy->non_zero_right != 0) {						
			for (int i = 0; i < 5; i++) {
				int bid_posit = int((tick->ask_bid[i].bid_price - tick->lower_limit_price) * cy->min_tick_div);
				int ask_posit = int((tick->ask_bid[i].ask_price - tick->lower_limit_price) * cy->min_tick_div);

				std::fill(cy->last_tick_row.begin() + bid_posit, cy->last_tick_row.begin() + ask_posit, 0);

				cy->last_tick_row[bid_posit] = tick->ask_bid[i].bid_volume;
				cy->last_tick_row[ask_posit] = tick->ask_bid[i].ask_volume;
				if (i = 4) {
					cy->non_zero_left = std::min(cy->non_zero_left, bid_posit);
					cy->non_zero_right = std::max(cy->non_zero_right, ask_posit);
				}
			}
		}
		else {
			for (int i = 0; i < 5; i++) {
				int bid_posit = int((tick->ask_bid[i].bid_price - tick->lower_limit_price) * cy->min_tick_div);
				int ask_posit = int((tick->ask_bid[i].ask_price - tick->lower_limit_price) * cy->min_tick_div);

				cy->last_tick_row[bid_posit] = tick->ask_bid[i].bid_volume;
				cy->last_tick_row[ask_posit] = tick->ask_bid[i].ask_volume;

				if (i = 4) {
					cy->non_zero_left = bid_posit;
					cy->non_zero_right = ask_posit;
				}
			}
		}

		for (int i = cy->non_zero_left; i <= cy->non_zero_right; i++) {
			(cy->record)[i][cy->tick_tail] = cy->last_tick_row[i];
		}
		// update tail position 
		cy->tick_tail == cy->n_tick - 1 ? cy->tick_tail = 0 : cy->tick_tail -= 1;
	}


	/*
		Return corresponding column with given price
	*/
	std::vector<int16_t> retrieve_ts_cylinder(struct Cylinder * cy, float price) {
		int posit = int((price - tick->lower_limit_price) * cy->min_tick_div);
		return (cy->record[posit]);
	}

	/*
		Build map from price to volume
	*/
	std::unordered_map<double, int> build_tick_volume(std::vector<NewOrder> const Tick *tick) {
		std::unordered_map<double, int> tick_volume;
		for (int i = 0; i < 5; i++) {
			tick_volume[tick->ask_bid[i].bid_price] = tick->ask_bid[i].bid_volume * -1;
			tick_volume[tick->ask_bid[i].ask_price] = tick->ask_bid[i].ask_volume;
		}
		return tick_volume;
	}

	/*
		Process new orders
	*/
	void match_new_orders(const vector<NewOrder> & new_order_list, std::unordered_map<int, Order> &order_list) {

		for (auto & new_order : new_order_list) {
			int sign = 0;
			if (new_order.quantity > 0){
				sign = 1;
			}
			else {
				sign = -1;
			}

			// Check for possible immediate match
			if ((new_order.quantity > 0 && new_order_price >= tick2->ask_bid[0].ask_price)
				|| (new_order.quantity < 0 && new_order_price <= tick2->ask_bid[0].bid_price)) {
				Order order_tmp = {new_order.order_id, new_order.price, new_order.quantity, 0, 0, sign};
				order_list[new_order.order_id] = order_tmp;
			}

			else if (tick_volume.find(new_order.price) != tick_volume.end()) {
				Order order_tmp = { new_order.order_id, new_order.price, new_order.quantity, tick2_volume[new_order.order_id],  tick2_volume[new_order.order_id], sign };
				order_list[new_order.order_id] = order_tmp;
			}
			else if (new_order.order_id > tick2->ask_bid[4].bid_price && new_order.order_id < tick2->ask_bid[4].ask_price) {
				Order order_tmp = { new_order.order_id, new_order.price, new_order.quantity, 0, 0, sign };
				order_list[new_order.order_id] = order_tmp;
			}
			else {
				// Here we assume no order out of current ob scope will be sent
				Order order_tmp = { new_order.order_id, new_order.price, new_order.quantity, -10000, -10000, sign };
				order_list[new_order.order_id] = order_tmp;
			}
		}
	}

	void match_old_orders(vector<Order> &order_list)
	{
		auto traded_dict = analyze_between_tick();

		if (tick1_volume == tick2_volume){
			return;
		}

		for (auto &order : order_list)
		{
			// Because we don't do the matching here, we only care about the following cases:
			// 1. When order has queue_posit = 0, we wait for done confirmation: so don't care gap and opposite prices

			if (order.queue_posit == 0) {
				continue;
			}

			// 2. When order was out of scope and now in scope for tick2
			else if (order.queue_posit == -10000) {
				if (tick2_volume.find(order.price) != tick2_volume.end()) {
					order.queue_posit = tick2_volume[order.price];
					order.queue_size = tick2_volume[order.price];
				}

			}

			// 3. Normal case, when price on
			else if (tick2_volume.find(order.price) != tick2_volume.end()) {
				int traded_volume = 0;
				if (traded_dict.find(order.price) != traded_dict.end()) {
					traded_volume = traded_dict[order.price];
				}

				int net_volume = std::abs(order.queue_size) - traded_volume - abs(tick2_volume[order.price]);
				int updated_posit = 0;
				if (net_volume <= 0) {
					updated_posit = std::abs(order.queue_posit) - traded_volume;
				}
				else {
					if (std::abs(order.queue_posit) * 2 <= std::abs(order.queue_size)) {
						updated_posit = std::abs(order.queue_posit) - traded_volume;
					}
					else {
						updated_posit = std::abs(order.queue_posit) - traded_volume - net_volume;
					}
				}
				if (updated_posit < 0) {
					order.queue_posit = 0;
					order.queue_size = tick2_volume[order.price];
				}
				else {
					updated_posit = std::min(updated_posit, std::abs(order.queue_posit)) * -1 * order.sign;
					order.queue_posit = updated_posit;
					order.queue_size = tick2_volume[order.price];
				}
			}
		}
	}

	/*

		Simple procedure to replicate between tick information
	*/
	unordered_map<double, double> analyze_between_tick()
	{
		unordered_map<double, double> ret;
		double volume = (tick2->total_volume - tick1->total_volume) / 2.0;  
		double cashVolume = ((tick1->total_turnover - tick2->total_turnover) / multiplier) / 2.0; 
		auto BidPrice = tick2->ask_bid[0].bid_price;
		auto AskPrice = tick2->ask_bid[0].ask_price;
		if (((int)BidPrice == 0 && (int)AskPrice == 0) || (volume <= 0.0) || (cashVolume < 0.0))
		{
			return ret;
		}
		else if ((int)BidPrice == 0)
		{
			ret[AskPrice] = volume;
			return ret;
		}
		else if ((int)AskPrice == 0)
		{
			ret[BidPrice] = volume;
			return ret;
		}

		//a = np.array([[tick2["BidPrice1"], tick2["AskPrice1"]], [1, 1]]) //#??D????¨®¡¤?3¨¬?¨°??D?¡À¨º¨¢?¡¤?3¨¬¡Á¨¦
		//b = np.array([cashVolume, volume])
		//x = np.linalg.solve(a, b)
		int x0 = (cashVolume - AskPrice * volume) / (BidPrice - AskPrice);
		int x1 = volume - x0;

		if (x0 < 0)
		{
			x0 = 0;
			x1 = volume;
		}
		else if (x1 < 0)
		{
			x0 = volume;
			x1 = 0;
		}

		ret[BidPrice] = x0;
		ret[AskPrice] = x1;
		return ret;
	}

	std::unordered_map<double, int> tick1_volume;
	std::unordered_map<double, int> tick2_volume;
	std::vector<NewOrder> Tick *tick1;
	std::vector<NewOrder> Tick *tick2;
	int multiplier;
	/*
		Variables/Objects defined here
	*/
	Cylinder obh;
}

int on_init(StrategyConfig *config) {
	//cout << "Using Matriqs" << endl;
	//bar::ma.min_tick = float(config->contracts[0].mini_move);
	foo::init_cylinder(&foo::obh, 240, 40, float(config->contracts[0].mini_move), 1);

	return 0;
}


void on_tick(Tick *tick) {
	cout << tick->symbol << " --------------------on-tick---------------------" << endl;

	/*
	if (bar::to_init) {
		//1 minutes
		bar::init_matriqs(&bar::ma, 240, 1, tick->lower_limit_price, tick->upper_limit_price);
		bar::to_init = false;
		return;
	}
	*/


	auto t1 = Clock::now();
	/*
	vector<float> tick_update = {
		float(tick->ask_bid[0].ask_price), float(tick->ask_bid[0].ask_volume),
		float(tick->ask_bid[1].ask_price), float(tick->ask_bid[1].ask_volume),
		float(tick->ask_bid[2].ask_price), float(tick->ask_bid[2].ask_volume),
		float(tick->ask_bid[3].ask_price), float(tick->ask_bid[3].ask_volume),
		float(tick->ask_bid[4].ask_price), float(tick->ask_bid[4].ask_volume),
		float(tick->ask_bid[0].bid_price), float(tick->ask_bid[0].bid_volume)*-1,
		float(tick->ask_bid[1].bid_price), float(tick->ask_bid[1].bid_volume)*-1,
		float(tick->ask_bid[2].bid_price), float(tick->ask_bid[2].bid_volume)*-1,
		float(tick->ask_bid[3].bid_price), float(tick->ask_bid[3].bid_volume)*-1,
		float(tick->ask_bid[4].bid_price), float(tick->ask_bid[4].bid_volume)*-1
	};
	*/
	vector<float> tick_update = {
		float(tick->ask_bid[0].ask_price), float(tick->ask_bid[0].ask_volume),
		float(tick->ask_bid[0].bid_price), float(tick->ask_bid[0].bid_volume)*-1
	};
	//bar::update_matriqs(&bar::ma, tick_update);
	foo::update_cylinder(&foo::obh, tick_update);
	auto t2 = Clock::now();

	for (auto i : tick_update) {
		std::cout << i << ' ';
	}
	std::cout << std::endl;
	std::cout << "Update time: "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
		<< " nanoseconds" << std::endl;

	auto t3 = Clock::now();
	//std::vector<int> histo = bar::retrieve_ts_matriqs(&bar::ma, float(tick->ask_bid[0].ask_price));
	std::vector<int16_t> histo = foo::retrieve_ts_cylinder(&foo::obh, float(tick->ask_bid[0].ask_price));
	auto t4 = Clock::now();

	for (auto i : histo) {
		std::cout << i << ' ';
	}
	std::cout << std::endl;
	std::cout << "Retrieve time: "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count()
		<< " nanoseconds" << std::endl;

}

void on_order_rsp(OrderResponse *order_response) {
	cout << order_response->symbol << " receive order order_response!" << endl;

	cout << "symbol: " << order_response->symbol << " "
		<< "local order id: " << order_response->local_order_id << " "
		<< "exchange order id: " << order_response->exchange_order_id << " "
		<< "close yday flag: " << order_response->close_yesterday_flag << " "
		<< "direction: " << order_response->direction << " "
		<< "openclose: " << order_response->openclose << " "
		<< "order voulme: " << order_response->order_volume << " "
		<< "order price£º" << order_response->order_price << " "
		<< "exec volume: " << order_response->executive_volume << " "
		<< "exec accumulated_executive_volume: " << order_response->accumulated_executive_volume << " "
		<< "order_status: " << order_response->order_status << " "
		<< "error id: " << order_response->error_id << " "
		<< "error msg: " << order_response->error_msg << endl;

	cout << order_response->symbol << "  pos: " << get_net_position(order_response->symbol) << endl;
}

void on_trade_rsp(TradeResponse *trade_response) {

	cout << trade_response->symbol << " receive trade trade_response!" << endl;
	cout << "symbol: " << trade_response->symbol << " "
		<< "local order id: " << trade_response->local_order_id << " "
		<< "exchange order id: " << trade_response->exchange_order_id << " "
		<< "exchange trade id: " << trade_response->exchange_trade_id << " "
		<< "order voulme: " << trade_response->order_volume << " "
		<< "order price£º" << trade_response->order_price << " "
		<< "exec voulme: " << trade_response->executive_volume << " "
		<< "exec price£º" << trade_response->executive_price << " "
		<< "exec accumulated_executive_volume: " << trade_response->accumulated_executive_volume << " "
		<< "close yday flag: " << trade_response->close_yesterday_flag << " "
		<< "direction: " << trade_response->direction << " "
		<< "openclose: " << trade_response->openclose << " "
		<< "order_status: " << trade_response->trade_status << endl;

	cout << trade_response->symbol << "  pos: " << get_net_position(trade_response->symbol) << endl;

}

void on_signal_int(void) {
	cout << "get the signal INT call~~~~~~~" << endl;

}

int on_set_parameters(const char *paramters) {
	//when you do not need to change the paramters in future, you just let this fucntion empty
	printf("\nset_param is called\n");
	printf("\nParamter is:%s\n", paramters);
	return 0;
}

int on_stop_strategy() {
	//this function will be called when the strategy stop outside, you may need to do something here,
	//such as cancel all the order unfilled
	printf("\non_stop_strategy is called\n");
	return 0;
}