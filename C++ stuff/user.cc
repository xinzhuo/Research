#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <map>
#include <unordered_map>
#include "user_space.h"
#include "user_quote_query.h"
#include "user_interface.h"
#include "json_parse_interface.h"
#include <string>
#include <time.h>
#include <math.h>
#include <array>
#include <list>
#include <inttypes.h>
#include <assert.h>

#define DZ 0.004
#define WL lhr_strategy::write_log_lhr
#define CURCP lhr_strategy::cmap[tick->symbol]
#define CURCP_ lhr_strategy::cmap[order_response->symbol]
#define CURCS lhr_strategy::cstatus[tick->symbol]
#define CURCS_ lhr_strategy::cstatus[order_response->symbol]
#define ADMETHOD 2
using std::string;
using std::cout;
using std::endl;
using std::vector;

namespace lhr_strategy
{
	class CS;
	struct CP;
	class TD;
	class CD;
	class QT;
	class TimeTest;

	class CS { //contract status
	public:
		char name[SYMBOL_LEN];
		bool has_exception;
		bool ready_to_execute;
		int real_position;

		double sprice;
		double trend;
		double risk;

		const CP *cp;
		TD *td;
		CD *cd;

		CS(CP *cp_, int rp);
		void show();
		void show_all();
	};
	struct CP { //contract property
		char name[SYMBOL_LEN];
		double step;
		double CdV;
		double inverse_multiplier;
		double inverse_min_step;
		bool close_yesterday;
		int valid_time[8];
		char main_contract[SYMBOL_LEN];
	};
	class TD { //tick data
	public:
		CS *cs;
		int size;
		bool ready;
		int position;
		vector<int64_t> total_volume;
		vector<int64_t> total_turnover;
		vector<double> ask_price_1;
		vector<double> bid_price_1;
		vector<uint32_t> ask_volume_1;
		vector<uint32_t> bid_volume_1;
		vector<bool> is_filled;
		double rise_limit;
		double fall_limit;

		double last_bid_price_1;
		double last_ask_price_1;

		TD(int size_, CS *cs_);
		void update_new_tick(Tick* tick);
		void fill_tick();
		void show();
	};
	class CD { //strategy calculate data
	public:
		CS *cs;
		TD *td;
		int size;								// Size of the vector
		int std_window;							// window size to calculate stdev, should be smaller than size
		double inverse_std_window;
		bool ready;
		int position;							// current location within the vector

		int counter;							// Count the number of tick

		// Custom defined variables
		vector<int64_t> tick_volume;
		vector<int64_t> tick_turnover;
		double tavp;

		vector<double> bid_rolling_sq_sum_list;
		vector<double> ask_rolling_sq_sum_list;
		
		double bid_rolling_sum;
		double ask_rolling_sum;
		double bid_rolling_sq_sum;
		double ask_rolling_sq_sum;

		double bid_stdev;						// stdev here is calculated exclude current tick, so it should be calculate on gap
		double ask_stdev;

		CD(int size_, CS *cs_);
		void movenext();
		void update();
		void show();
		void update_movenext();
	};
	class QT { //quote ticks
	public:
		int size;
		bool ready;
		int position;
		uint64_t last_time;
		uint64_t last_gap;
		vector<TD*> tds;
		vector<CD*> cds;

		QT(int size_);
		//void autofill_end();
		//void autocompletion_end();
		void autocompletion_head();
		//void new_head();
		bool check();
		void show();
		void show_all();
	};
	class TimeTest {
	private:
		std::array<uint64_t, 7> timearray;
		char name[SYMBOL_LEN];

	public:
		void conclude();
		void get_tick_time(Tick *tick);
		void record(int i);
		void clear() {
			timearray.fill(0);
		}
		TimeTest() {
			timearray.fill(0);
		}
	};
	namespace SP {
		const static int storelen = 200;										//Vector size defined here
	};

	const bool is_simulating = true;
	const bool is_debugging = true;
	const bool is_724 = false;

	std::unordered_map<string, CS*> cstatus;
	std::unordered_map<string, CP*> cmap;
	TimeTest tt;
	QT qt(SP::storelen);

	uint64_t get_system_time_nsec()
	{
		struct timespec time = { 0, 0 };
		clock_gettime(CLOCK_REALTIME, &time);
		return time.tv_sec * 1000000000 + time.tv_nsec;
	}
	void write_log_lhr(const char *log) {
		if (is_debugging) {
			cout << log << endl;
		}
		write_log(log);
	}
	void write_log_fatal(const char *log) {
		cout << log << endl;
		write_log(log);
	}

	CS::CS(CP *cp_, int rp) {
		cp = cp_;
		this->has_exception = false;
		this->ready_to_execute = false;
		this->real_position = rp;
		this->td = new lhr_strategy::TD(lhr_strategy::SP::storelen, this);
		this->cd = new lhr_strategy::CD(lhr_strategy::SP::storelen, this);
	}
	void CS::show() {
		char s[512];
		sprintf(s, "[cs_info] name:%s real_position:%d has_exception:%d", name, real_position, has_exception);
		WL(s);
	}
	void CS::show_all() {
		this->show();
		td->show();
		cd->show();
	}

	TD::TD(int size_, CS *cs_) {
		cs = cs_;
		size = size_;
		total_volume.resize(size, 0);
		total_turnover.resize(size, 0);
		ask_price_1.resize(size, 0);
		bid_price_1.resize(size, 0);
		ask_volume_1.resize(size, 0);
		bid_volume_1.resize(size, 0);
		is_filled.resize(size, 0);
		rise_limit = fall_limit = NAN;
		ready = false;
		position = -1;

		last_bid_price_1 = 0.0;
		last_ask_price_1 = 0.0;
	}
	void TD::update_new_tick(Tick *tick) { //on tick
		if ((this->position < 0 && this->ready) || this->position >= this->size) {
			printf("[td_error] %s position out of range", tick->symbol);
			exit(-1);
		}
		this->ask_price_1[this->position] = tick->ask_bid[0].ask_price;
		this->bid_price_1[this->position] = tick->ask_bid[0].bid_price;
		this->ask_volume_1[this->position] = tick->ask_bid[0].ask_volume;
		this->bid_volume_1[this->position] = tick->ask_bid[0].bid_volume;
		this->total_volume[this->position] = tick->total_volume;
		this->total_turnover[this->position] = tick->total_turnover;
		this->rise_limit = tick->upper_limit_price;
		this->fall_limit = tick->lower_limit_price;
		

		if (this->is_filled[this->position]) {
			this->is_filled[this->position] = false;
		}
		else {
			WL("[td_info] tick data overwrite.");
		}
	}
	void TD::fill_tick() { //on_gap
		if ((this->position < 0 && this->ready) || this->position >= this->size) {
			printf("[td_error] %s position out of range", cs->name);
			exit(-1);
		}
		int last_position = this->position;
		this->position += 1;
		if (this->position >= this->size) {
			this->position -= this->size;
			if (!this->ready) {
				this->ready = true;
			}
		}
		this->ask_price_1[this->position] = this->ask_price_1[last_position];
		this->bid_price_1[this->position] = this->bid_price_1[last_position];
		this->ask_volume_1[this->position] = this->ask_volume_1[last_position];
		this->bid_volume_1[this->position] = this->bid_volume_1[last_position];
		this->total_volume[this->position] = this->total_volume[last_position];
		this->total_turnover[this->position] = this->total_turnover[last_position];
		this->is_filled[this->position] = true;

		// Self defined
		this->last_bid_price_1 = this->bid_price_1[last_position];
		this->last_ask_price_1 = this->ask_price_1[last_position];
	}
	void TD::show() {
		bool filled = is_filled[position];
		char s[512];
		sprintf(s, "[td_info] name:%s size:%d, ready:%d position:%d ask_price_1:%f bid_price_1:%f ask_volume_1:%d bid_volume_1:%d last_filled:%d rise_limit:%f fall_limit:%f",
			cs->name, size, ready, position, ask_price_1[position], bid_price_1[position], ask_volume_1[position], bid_volume_1[position], filled, rise_limit, fall_limit);
		WL(s);
	}

	CD::CD(int size_, CS *cs_) {
		cs = cs_;
		td = cs->td;
		size = size_;
		ready = false;
		position = -1;
		counter = 0;
		std_window = 20;
		inverse_std_window = 1.0 / std_window;
		assert(std_window < size);
		tavp = 0;
		bid_rolling_sum = 0;
		ask_rolling_sum = 0;

		bid_rolling_sq_sum = 0;
		ask_rolling_sq_sum = 0;
		bid_stdev = 0;
		ask_stdev = 0;

		tick_volume.resize(size, 0);
		tick_turnover.resize(size, 0);
		bid_rolling_sq_sum_list.resize(size, 0);
		ask_rolling_sq_sum_list.resize(size, 0);
	}
	void CD::movenext() { //on gap
		this->position += 1;
		if (this->position >= this->size) {
			this->position -= this->size;
			if (!this->ready) {
				this->ready = true;
			}
		}
		this->update_movenext();

		return;
	}
	void CD::update_movenext() {
		// bid_stdev, ask_stdev, excluding current tick
		// Position referred to the position in the TD vectors
		int previous_n_position = this->position - this->std_window;
		if (this->ready && previous_n_position < 0) {
			previous_n_position += this.size;
		}
		int last_position = this->position - 1;
		if (this->ready && last_position < 0) {
			last_position += this->size;
		}
		// We update bid/ask_rolling_sq_sum_list here, because we only use last tick data,
		this->bid_rolling_sq_sum_list[last_position] = this->td->bid_price_1[last_position] * this->td->bid_price_1[last_position];
		this->ask_rolling_sq_sum_list[last_position] = this->td->ask_price_1[last_position] * this->td->ask_price_1[last_position];
		
		// -20, -19 ... -1
		this->bid_rolling_sum = this->bid_rolling_sum - this->td->bid_price_1[previous_n_position] + this->td->bid_price_1[last_position];
		this->ask_rolling_sum = this->ask_rolling_sum - this->td->ask_price_1[previous_n_position] + this->td->ask_price_1[last_position];
		this->bid_rolling_sq_sum = this->bid_rolling_sq_sum - this->bid_rolling_sq_sum_list[previous_n_position] + this->bid_rolling_sq_sum_list[last_position];
		this->ask_rolling_sq_sum = this->ask_rolling_sq_sum - this->ask_rolling_sq_sum_list[previous_n_position] + this->ask_rolling_sq_sum_list[last_position];

		this->bid_stdev = sqrt(this->bid_rolling_sq_sum * this->inverse_std_window - pow(this->bid_rolling_sum * this->inverse_std_window, 2));
		this->ask_stdev = sqrt(this->ask_rolling_sq_sum * this->inverse_std_window - pow(this->ask_rolling_sum * this->inverse_std_window, 2));
	}

	void CD::update() { // on tick
		this->counter += 1;
		int last_position = this->position - 1;
		if (this->ready && last_position < 0) {
			last_position += this->size;
		}
		if (this->position != this->td->position) {
			printf("position in cd do not fit position in td");
			exit(-1);
		}

		//tick_turnover and tick_volume
		if (last_position == -1) {
			this->tick_turnover[this->position] = this->tick_volume[this->position] = 0;
		}
		else {
			this->tick_turnover[this->position] = this->td->total_turnover[this->position] - this->td->total_turnover[last_position];
			this->tick_volume[this->position] = this->td->total_volume[this->position] - this->td->total_volume[last_position];
		}

		//tavp
		if (this->tick_volume[this->position] == 0) {
			this->tavp = NAN;
		}
		else {
			this->tavp = this->tick_turnover[this->position] / this->tick_volume[this->position] * cs->cp->inverse_multiplier;
		}
		return;
	}
	void CD::show() {
		char s[512];
		sprintf(s, "[cd_info] name:%s size:%d ready:%d position:%d td_position:%d tavp:%f  tick_volume:%" PRId64 " tick_turnover:%" PRId64,
			cs->name, size, ready, position, td->position, tavp, tick_volume[position], tick_turnover[position]);
		WL(s);
	}

	QT::QT(int size_) {
		size = size_;
		ready = false;
		position = -1;
		last_time = 0;
		last_gap = 0;
	}
	void QT::autocompletion_head() {
		//this->check();
		if (this->check()) {
			printf("position in tds and cds do not fit position in qt before autocompletion");
			exit(-1);
		}
		this->position += 1;
		if (this->position >= this->size) {
			this->position -= this->size;
			if (!this->ready) {
				this->ready = true;
			}
		}
		for (auto it = tds.begin(); it != tds.end(); it++) {
			(*it)->fill_tick();
		}
		for (auto it = cds.begin(); it != cds.end(); it++) {
			(*it)->movenext();
		}
		//this->check();
		if (this->check()) {
			printf("position in tds and cds do not fit position in qt after autocompletion");
			exit(-1);
		}
		uint64_t lt = last_time;
		last_time = get_system_time_nsec();
		last_gap = last_time - lt;
	}
	bool QT::check() {
		for (auto it = tds.begin(); it != tds.end(); it++) {
			if ((*it)->position != this->position) {
				return true;
			}
		}
		for (auto it = cds.begin(); it != cds.end(); it++) {
			if ((*it)->position != this->position) {
				return true;
			}
		}
		return false;
	}
	void QT::show() {
		char s[512];
		sprintf(s, "[qt_info] size:%d ready:%d position:%d",
			size, ready, position);
		WL(s);
	}
	void QT::show_all() {
		this->show();
		for (auto it = this->tds.begin(); it != this->tds.end(); it++) {
			(*it)->cs->show_all();
		}
	}

	void TimeTest::conclude() {
		char s[512];
		sprintf(s, "[time info] %s on tick time: %" PRId64 " part times: %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "; total time: %" PRId64 "",
			this->name,
			timearray[1],
			timearray[1] - timearray[0],
			timearray[2] - timearray[1],
			timearray[3] - timearray[2],
			timearray[4] - timearray[3],
			timearray[5] - timearray[4],
			timearray[6] - timearray[5],
			timearray[6] - timearray[0]);
		write_log_lhr(s);
		return;
	}
	void TimeTest::get_tick_time(Tick *tick) {
		strcpy(this->name, tick->symbol);
		timearray[0] = tick->system_time_ns;
	}
	void TimeTest::record(int i) {
		timearray[i] = get_system_time_nsec();
	}

	void strategy_prepare(CS *cs) { //strategy, generate expect position and expect price or priority
		if (round((cs->td->last_bid_price_1 - cs->td->bid_price_1[cs->td->position]) * cs->cp->inverse_min_step) >= 4
			&& cs->td->last_ask_price_1 == cs->td->ask_price_1[cs->td->position]) {
			if (cs->cd->tick_volume[cs->cd->position] == 0 ||
				(cs->cd->tavp >= cs->td->bid_price_1[cs->td->position] &&
					cs->cd->ask_stdev < 2 * cs->cp->step)) {
				cout << "Buy" << endl;
				}
			}
		else if (round((cs->td->ask_price_1[cs->td->position] - cs->td->last_ask_price_1) * cs->cp->inverse_min_step) >= 4 
			&& cs->td->last_bid_price_1 == cs->td->bid_price_1[cs->td->position]) {
			if (cs->cd->tick_volume[cs->cd->position] == 0 || (cs->cd->tavp <= cs->td->ask_price_1[cs->td->position] && cs->cd->bid_stdev < 2 * cs->cp->step))
			{
				cout << "Sell" << endl;
			}
		}
	}

	void strategy_execute(CS *cs) { //execute stategy

	}
}

int on_init(StrategyConfig *config) {
	for (uint32_t i = 0; i < config->contracts_num; ++i) {
		cout << "----------------------on_init of " << "<config->contracts[i].symbol"
			<< "-----------------------" << endl;
		cout << "symbol: " << config->contracts[i].symbol << ", "
			<< "account: " << config->contracts[i].account << ", "
			<< "max pos: " << config->contracts[i].max_pos << ", "
			<< "max single_side_max_pos: " << config->contracts[i].single_side_max_pos << ", "
			<< "max_accum_open_vol:" << config->contracts[i].max_accum_open_vol << ","
			<< "expiration_date:" << config->contracts[i].expiration_date << ", "
			<< "multipe: " << config->contracts[i].multipe << ", "
			<< "cancel_times:" << config->contracts[i].cancel_times << ","
			<< "max_cancel_times:" << config->contracts[i].max_cancel_times << ","
			<< "mini_move: " << config->contracts[i].mini_move << endl;
		printf("today pos: [%u %lf %u %lf]\n",
			config->contracts[i].today_pos.long_volume,
			config->contracts[i].today_pos.long_price,
			config->contracts[i].today_pos.short_volume,
			config->contracts[i].today_pos.short_price);

		printf("yesterday pos: [%u %lf %u %lf]\n",
			config->contracts[i].yesterday_pos.long_volume,
			config->contracts[i].yesterday_pos.long_price,
			config->contracts[i].yesterday_pos.short_volume,
			config->contracts[i].yesterday_pos.short_price);

		string sc_string = config->contracts[i].symbol;
		string s_string, c_string;
		for (int i = 0; i < 32; i++) {
			if (sc_string[i] >= '0' && sc_string[i] <= '9') {
				s_string = sc_string.substr(0, i);
				c_string = sc_string.substr(i);
				break;
			}
		}
		cout << "symbol:" << s_string << endl;

		lhr_strategy::cmap[config->contracts[i].symbol] = new lhr_strategy::CP;
		lhr_strategy::CP *this_cp = lhr_strategy::cmap[config->contracts[i].symbol];
		strcpy(this_cp->name, config->contracts[i].symbol);
		this_cp->step = config->contracts[i].mini_move;
		this_cp->CdV = config->contracts[i].multipe;
		this_cp->inverse_min_step = 1 / config->contracts[i].mini_move;
		this_cp->inverse_multiplier = 1 / config->contracts[i].multipe;
		

		if (s_string == "IF" || s_string == "IC" || s_string == "IH") {
			this_cp->valid_time[0] = 93200000;
			this_cp->valid_time[1] = 112800000;
			this_cp->valid_time[2] = 93200000;
			this_cp->valid_time[3] = 112800000;
			this_cp->valid_time[4] = 130200000;
			this_cp->valid_time[5] = 145800000;
			this_cp->valid_time[6] = 130200000;
			this_cp->valid_time[7] = 145800000;
		}
		else if (s_string == "T" || s_string == "TH") {
			this_cp->valid_time[0] = 91700000;
			this_cp->valid_time[1] = 112800000;
			this_cp->valid_time[2] = 91700000;
			this_cp->valid_time[3] = 112800000;
			this_cp->valid_time[4] = 130200000;
			this_cp->valid_time[5] = 151300000;
			this_cp->valid_time[6] = 130200000;
			this_cp->valid_time[7] = 151300000;
		}
		else {
			this_cp->valid_time[0] = 90200000;
			this_cp->valid_time[1] = 101300000;
			this_cp->valid_time[2] = 103100000;
			this_cp->valid_time[3] = 112800000;
			this_cp->valid_time[4] = 133200000;
			this_cp->valid_time[5] = 145800000;
			if (s_string == "pp" || s_string == "v" || s_string == "l" || s_string == "bb" || s_string == "fb" ||
				s_string == "c" || s_string == "cs" || s_string == "jd" || s_string == "SM" || s_string == "SF" ||
				s_string == "WH" || s_string == "JR" || s_string == "LR" || s_string == "PM" || s_string == "RI" ||
				s_string == "RS" || s_string == "AP" || s_string == "fu" || s_string == "wr") {
				this_cp->valid_time[6] = 133200000;
				this_cp->valid_time[7] = 145800000;
			}
			else if (s_string == "i" || s_string == "j" || s_string == "jm" || s_string == "a" || s_string == "b" ||
				s_string == "m" || s_string == "p" || s_string == "y" || s_string == "FG" || s_string == "MA" ||
				s_string == "SR" || s_string == "TA" || s_string == "RM" || s_string == "OI" || s_string == "CF" ||
				s_string == "CY" || s_string == "ZC") {
				this_cp->valid_time[6] = 210200000;
				this_cp->valid_time[7] = 232800000;
			}
			else if (s_string == "ru" || s_string == "bu" || s_string == "rb" || s_string == "hc" ||
				s_string == "sp") {
				this_cp->valid_time[6] = 210200000;
				this_cp->valid_time[7] = 225800000;
			}
			else if (s_string == "cu" || s_string == "pb" || s_string == "al" || s_string == "zn" ||
				s_string == "sn" || s_string == "ni") {
				this_cp->valid_time[6] = 210200000;
				this_cp->valid_time[7] = 245800000;
			}
			else if (s_string == "au" || s_string == "ag") {
				this_cp->valid_time[6] = 210200000;
				this_cp->valid_time[7] = 262800000;
			}
			else {
				lhr_strategy::write_log_fatal("[on_init_fatal_error] unknown contract name");
				exit(-1);
			}
		}
		if (s_string == "IF" || s_string == "IC" || s_string == "IH") {
			strcpy(this_cp->main_contract, (s_string + "1901").c_str());
		}
		else if (s_string == "T" || s_string == "TH") {
			strcpy(this_cp->main_contract, (s_string + "1903").c_str());
		}
		else if (s_string == "m" || s_string == "y" || s_string == "a" || s_string == "b" || s_string == "p" ||
			s_string == "c" || s_string == "cs" || s_string == "jd" || s_string == "l" || s_string == "v" ||
			s_string == "pp" || s_string == "j" || s_string == "jm" || s_string == "i") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else if (s_string == "eg") {
			strcpy(this_cp->main_contract, (s_string + "1906").c_str());
		}
		else if (s_string == "cu" || s_string == "pb" || s_string == "al" || s_string == "zn" ||
			s_string == "sn") {
			strcpy(this_cp->main_contract, (s_string + "1903").c_str());
		}
		else if (s_string == "ni" || s_string == "sn" || s_string == "fu" || s_string == "ru") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else if (s_string == "au" || s_string == "ag" || s_string == "bu") {
			strcpy(this_cp->main_contract, (s_string + "1906").c_str());
		}
		else if (s_string == "rb" || s_string == "hc") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else if (s_string == "SR" || s_string == "CF" || s_string == "FG" || s_string == "MA" || s_string == "OI" ||
			s_string == "RM" || s_string == "SF" || s_string == "SM") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else if (s_string == "ZC" || s_string == "TA") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else if (s_string == "AP") {
			strcpy(this_cp->main_contract, (s_string + "1905").c_str());
		}
		else {
			lhr_strategy::write_log_fatal("[on_init_fatal_error] unknown contract name to get main contract");
			exit(-1);
		}

		int rp = config->contracts[i].yesterday_pos.long_volume + config->contracts[i].today_pos.long_volume -
			config->contracts[i].yesterday_pos.short_volume - config->contracts[i].today_pos.short_volume;
		lhr_strategy::cstatus[config->contracts[i].symbol] = new lhr_strategy::CS(this_cp, rp);
		lhr_strategy::CS *this_cs = lhr_strategy::cstatus[config->contracts[i].symbol];
		strcpy(this_cp->name, config->contracts[i].symbol);
		strcpy(this_cs->name, config->contracts[i].symbol);
		
		lhr_strategy::qt.tds.push_back(this_cs->td);
		lhr_strategy::qt.cds.push_back(this_cs->cd);

		if (s_string == "j" || s_string == "jm") {
			if (c_string.substr(2) == "01" || c_string.substr(2) == "05" || c_string.substr(2) == "09") {
				cout << "set " << config->contracts[i].symbol << "'s close_yesterday true";
				this_cp->close_yesterday = true;
			}
			else {
				cout << "set " << config->contracts[i].symbol << "'s close_yesterday false";
				this_cp->close_yesterday = false;
			}
		}
		else if (s_string == "bu") {
			if (c_string == "1806") {
				cout << "set " << config->contracts[i].symbol << "'s close_yesterday true";
				this_cp->close_yesterday = true;
			}
		}
		else if (s_string == "IC" || s_string == "IF" || s_string == "IH") {
			cout << "set " << config->contracts[i].symbol << "'s close_yesterday true";
			this_cp->close_yesterday = true;
		}
		else {
			cout << "set " << config->contracts[i].symbol << "'s close_yesterday false";
			this_cp->close_yesterday = false;
		}
	}
	
	cout << "contracts_num: " << config->contracts_num << ", "
		<< "strategy_id: " << config->strategy_id << ", "
		<< "strategy_name: " << config->strategy_name << ", "
		<< "day_night_flag: " << config->day_night_flag << ", "
		<< "strategy_param_file: " << config->strategy_param_file << ", "
		<< "file_path: " << config->file_path << ", "
		<< "so_path: " << config->so_path << ","
		<< "account: " << config->accounts[0].account << endl;
	lhr_strategy::qt.autocompletion_head();
	lhr_strategy::qt.show();
	return 0;
}

void on_tick(Tick *tick) {
	lhr_strategy::tt.record(1);
	lhr_strategy::tt.get_tick_time(tick);
	lhr_strategy::CS *cur_cs = CURCS;

	lhr_strategy::tt.record(2);
	//maintain datas need to be store
	char s[512];
	sprintf(s, "%f %f %d %d", tick->ask_bid[0].ask_price, tick->ask_bid[0].bid_price, tick->ask_bid->ask_volume, tick->ask_bid->bid_volume);
	WL(s);
	cur_cs->td->update_new_tick(tick);
	cur_cs->cd->update();
	lhr_strategy::tt.record(3);
	if (cur_cs->has_exception) {
		char s[512];
		sprintf(s, "[info] %s has exception, skip", tick->symbol);
		WL(s);
		return;
	}
	lhr_strategy::tt.record(4);
	lhr_strategy::strategy_prepare(cur_cs);
	lhr_strategy::tt.record(5);
	lhr_strategy::strategy_execute(cur_cs);
	lhr_strategy::tt.record(6);
	lhr_strategy::tt.conclude();
	lhr_strategy::tt.clear();
	return;
}

void on_gap() {
	uint64_t time_ns = lhr_strategy::get_system_time_nsec();
	char s[512];
	sprintf(s, "[on_gap_time] %" PRIu64 " gap: %" PRIu64 "", time_ns, lhr_strategy::qt.last_gap);
	WL(s);
	if (!lhr_strategy::is_simulating) {
		if (lhr_strategy::qt.last_gap > 600000000) {
			lhr_strategy::write_log_fatal("[gap_time_warning] largger than 600ms");
		}
		else if (lhr_strategy::qt.last_gap < 400000000) {
			lhr_strategy::write_log_fatal("[gap_time_warning] smaller than 400ms");
		}
	}
	lhr_strategy::qt.show_all();
	lhr_strategy::qt.autocompletion_head();		//Call movenext
}

void on_order_rsp(OrderResponse *order_response) {
	return;
}

void on_trade_rsp(TradeResponse *order_response) {
	return;
}

int on_set_parameters(const char *parameters) {
	return 0;
}

int on_stop_strategy() {
	return 0;
}

void on_signal_int(void) {
	return;
}