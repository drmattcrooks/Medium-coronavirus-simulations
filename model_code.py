import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.graph_objects as go
import tqdm
import numpy.matlib

class disease_model_class(object):

    def __init__(self,
                 age_data_path='../data/age_population.csv',
                 number_of_people=200,
                 social_distance=False,
                 scale=0.01,
                 add_immunity=False,
                 shield_over70s=False,
                 close_schools=False,
                 isolate_sick=False,
                 include_asymptomatic=False,
                 scenario=None):

        self.scenario_dict = {'F': False, 'T': True}
        self._use_exisiting_xy = False

        scenarios = ['T', 'F']
        for rep in range(5):
            scenarios = [scen + 'T' for scen in scenarios] + [scen + 'F' for scen in scenarios]
        self.scenario_id_dict = dict(zip(scenarios, range(len(scenarios))))

        if scenario is not None:
            self._set_scenario(scenario)
            self.scenario = scenario
        else:
            self.add_immunity = add_immunity
            self.shield_over70s = shield_over70s
            self.close_schools = close_schools
            self.isolate_sick = isolate_sick
            self.include_asymptomatic = include_asymptomatic
            self.social_distance = social_distance
            self.scenario = self._get_scenario_id()

        np.random.seed(38)

        dfages = pd.read_csv(age_data_path)
        self.ages_i = dfages['i'].values
        self.p_i = dfages['p_i'].values * 1000000
        self.H_i = dfages['H_i'].values
        self.h_i = dfages['h_i'].values
        self.d_i = dict(zip(self.ages_i, 2 * self.h_i))

        self.number_of_people = number_of_people


        self.θ = 3
        self.hours_in_a_day = 12
        self.days_until_cured = (5 + 7) * self.hours_in_a_day
        self.loc = 0
        self.scale = scale

        # fraction of all asymptomatic cases - split across ages proportionally to the hospitalisation fraction
        γ = 10
        symptom_rate = γ * self.h_i
        symptom_rate[symptom_rate > 1] = 1
        symptom_rate_dict = dict(zip(self.ages_i, symptom_rate))


        self.ages = np.random.choice(a=self.ages_i,
                                     p=self.p_i / sum(self.p_i),
                                     size=self.number_of_people)

        self.is_symptomatic = []
        for person, age in enumerate(self.ages):
            βi = symptom_rate_dict[age]
            self.is_symptomatic.append(np.random.choice(a=[1, 0],
                                                        p=[βi, 1 - βi],
                                                        size=1)[0])
        # plot color dict
        self.infection_dict = {0: 'not_infected',
                                        1: 'infected'}
        self.person_colors = {0: 'steelblue',
                                       1: 'firebrick'}

        self.symptoms_start = 5 * self.hours_in_a_day

        self.shield_ages = ['70-74', '75-79', '80-84', '85-90', '90+']
        self.school_ages = ['0-4', '5-9', '10-14', '15-19']

    def _update_locations(self):

        self.x = self.x + np.random.normal(loc=0, scale=self.scale, size=self.number_of_people) \
             * (1 - self.deceased) * (1 - self.is_shielding) * (1 - self.is_isolating)
        self.y = self.y + np.random.normal(loc=0, scale=self.scale, size=self.number_of_people) \
             * (1 - self.deceased) * (1 - self.is_shielding) * (1 - self.is_isolating)

        self.y[self.y > 1] = 2 - self.y[self.y > 1]
        self.y[self.y < 0] = - self.y[self.y < 0]
        self.x[self.x > 1] = 2 - self.x[self.x > 1]
        self.x[self.x < 0] = - self.x[self.x < 0]

    def _spread_infections(self):
        dist = np.zeros((self.number_of_people, self.number_of_people))
        for person in range(self.number_of_people):
            dist[person, :] = np.sqrt((self.x - self.x[person]) ** 2 + (self.y - self.y[person]) ** 2)

        # Who are close enough to pass infection
        close_together_people = 1 * (dist < self.contamination_distance)

        # cooccurence infection among each pair of people
        infected_delta = np.matlib.repmat(self.infected, self.number_of_people, 1)
        infected_delta = 1 * ((infected_delta + infected_delta.T) > 0)

        # Pass infection between infected people and people they are close enough to
        self.infected = 1 * ((close_together_people * infected_delta).sum(axis=1) > 0)
        if self.add_immunity:
            # remove people with immunity
            self.infected[self.immune == 1] = 0

        self.infected = self.infected * (1 - self.is_shielding) * (1 - self.deceased)

    # count number of days infected
    def _update_days_infected(self):
        self.days_infected = self.days_infected + self.infected

    # remove cured people
    def _cure_people(self):

        self.infected[self.days_infected == self.days_until_cured] = 0
        if self.add_immunity:
            self.immune[self.days_infected == self.days_until_cured] = 1

        for i, day in enumerate(self.days_infected):
            if self.days_infected[i] == self.days_until_cured:
                self.deceased[i] = np.random.choice(a=[0, 1],
                                                    p=[1 - self.d_i[self.ages[i]],
                                                       self.d_i[self.ages[i]]])

        if self.isolate_sick:
            self.is_isolating[self.days_infected == self.symptoms_start] = 1
            self.is_isolating = self.is_isolating * self.is_symptomatic
            self.is_isolating[self.days_infected == self.days_until_cured] = 0

        self.days_infected[self.days_infected == self.days_until_cured] = 0

    def _initiate_simulation(self):

        if self._use_exisiting_xy:
            self.x = self.dfplot.loc[self.dfplot['day_ind'] == 0, 'x']
            self.y = self.dfplot.loc[self.dfplot['day_ind'] == 0, 'y']
            self.infected = self.dfplot.loc[self.dfplot['day_ind'] == 0, 'infected']
        else:
            self.x = np.random.uniform(0, 1, self.number_of_people)
            self.y = np.random.uniform(0, 1, self.number_of_people)
            if self.θ < 1:
                self.infected = np.random.choice(a=[0, 1], p=[1 - self.θ, self.θ], size=self.number_of_people)
            else:
                self.infected = [1 for i in range(self.θ)] + [0 for i in range(self.number_of_people - self.θ)]
                np.random.shuffle(self.infected)

        self.days_infected = self.infected.copy()
        self.immune = np.zeros(self.number_of_people)
        self.deceased = np.zeros(self.number_of_people)
        self.is_isolating = np.zeros(self.number_of_people)

        self.is_shielding = np.zeros(self.number_of_people)
        if self.shield_over70s:
            self.is_shielding = np.array([age in self.shield_ages for age in self.ages])
        if self.close_schools:
            self.is_shielding = self.is_shielding + np.array([age in self.school_ages for age in self.ages])
        if self.include_asymptomatic is False:
            self.is_symptomatic = np.ones(self.number_of_people)

        if self.social_distance:
            self.contamination_distance = 0.015
        else:
            self.contamination_distance = 0.03

        self.dfplot = pd.DataFrame(columns=['x', 'y', 'color', 'day', 'infected', 'person_id', 'immune', 'deceased'])
        self.dfplot['x'] = self.x
        self.dfplot['y'] = self.y
        self.dfplot['color'] = [self.person_colors[inf_] for inf_ in self.infected]
        self.dfplot['day_ind'] = 0
        self.dfplot['size'] = 2
        self.dfplot['person_id'] = range(self.number_of_people)
        self.dfplot['infected'] = self.infected
        self.dfplot['infection_status'] = [self.infection_dict[inf_] for inf_ in self.infected]
        self.dfplot['immune'] = 0
        self.dfplot['deceased'] = 0

    def _set_scenario(self, scenario):

        self.add_immunity = self.scenario_dict[scenario[0]]
        self.include_asymptomatic = self.scenario_dict[scenario[1]]
        self.isolate_sick = self.scenario_dict[scenario[2]]
        self.social_distance = self.scenario_dict[scenario[3]]
        self.shield_over70s = self.scenario_dict[scenario[4]]
        self.close_schools = self.scenario_dict[scenario[5]]

        if self.social_distance:
            self.contamination_distance = 0.015
        else:
            self.contamination_distance = 0.03

    def _create_scenario_id(self,
                            add_immunity=False,
                            include_asymptomatic=False,
                            isolate_sick=False,
                            social_distance=False,
                            shield_over70s=False,
                            close_schools=False):

        id = 'FFFFFF'
        if add_immunity:
            id = 'T' + id[1:]
        if include_asymptomatic:
            id = id[:1] + 'T' + id[2:]
        if isolate_sick:
            id = id[:2] + 'T' + id[3:]
        if social_distance:
            id = id[:3] + 'T' + id[4:]
        if shield_over70s:
            id = id[:4] + 'T' + id[5:]
        if close_schools:
            id = id[:5] + 'T'
        return id

    def _get_scenario_id(self):

        id = 'FFFFFF'
        if self.add_immunity:
            id = 'T' + id[1:]
        if self.include_asymptomatic:
            id = id[:1] + 'T' + id[2:]
        if self.isolate_sick:
            id = id[:2] + 'T' + id[3:]
        if self.social_distance:
            id = id[:3] + 'T' + id[4:]
        if self.shield_over70s:
            id = id[:4] + 'T' + id[5:]
        if self.close_schools:
            id = id[:5] + 'T'
        return id

    def run_simulation(self, timesteps, save_separate_output=False):
        self.timesteps = timesteps

        self.save_separate_output = save_separate_output

        self._initiate_simulation()
        for t in range(1, timesteps):

            self._update_locations()
            self._spread_infections()
            self._update_days_infected()
            self._cure_people()

            df_ = pd.DataFrame(columns=['x', 'y', 'color', 'day', 'size', 'infected', 'person_id', 'immune'])
            df_['x'] = self.x
            df_['y'] = self.y
            df_['color'] = [self.person_colors[inf_] for inf_ in self.infected]
            df_['day_ind'] = t
            df_['size'] = 2
            df_['person_id'] = range(self.number_of_people)
            df_['infected'] = self.infected
            # df_['infection_status'] = [self.infection_dict[inf_] for inf_ in self.infected]
            df_['immune'] = self.immune
            df_['deceased'] = self.deceased
            df_.loc[df_['immune'] == 1, 'infected'] = 2
            df_.loc[df_['deceased'] == 1, 'infected'] = 3
            self.dfplot = pd.concat((self.dfplot, df_), sort=False)

        if self.save_separate_output:
            scenario_id = self._create_scenario_id(
                add_immunity=self.add_immunity,
                include_asymptomatic=self.include_asymptomatic,
                isolate_sick=self.isolate_sick,
                social_distance=self.social_distance,
                shield_over70s=self.shield_over70s,
                close_schools=self.close_schools)
            self.dfplot.to_csv(f'../data/all_scenarios/scenario_{scenario_id}.csv', index=False)

    def run_multiple_scenarios(self, timesteps, scenario_list):

        self._use_exisiting_xy = False

        number_of_trials = len(scenario_list)
        with tqdm.tqdm(total=number_of_trials) as progress:
            for scenario in scenario_list:
                scenario_id = self.scenario_id_dict[scenario]

                self._set_scenario(scenario)
                self.run_simulation(timesteps)
                if self.save_separate_output:
                    if self._use_exisiting_xy == False:
                        self.dfplot_multi = self.dfplot[
                            ['x', 'y', 'day', 'person_id', 'infected']].rename(
                            columns={'x': 'x_' + str(scenario_id),
                                     'y': 'y_' + str(scenario_id),
                                     'infected': 'infected_' + str(scenario_id)}).copy()
                        self._use_exisiting_xy = True
                    else:
                        self.dfplot_multi = pd.merge(self.dfplot_multi,
                                               self.dfplot[['x', 'y', 'day', 'person_id', 'infected']].rename(
                                                   columns={'x': 'x_' + str(scenario_id),
                                                            'y': 'y_' + str(scenario_id),
                                                            'infected': 'infected_' + str(scenario_id)}),
                                               how='inner',
                                               on=['day', 'person_id'])
                progress.update(1)

    def run_all_scenarios(self,
                          timesteps,
                          save_separate_output=False,
                          save_output=True):

        self.save_separate_output = save_separate_output

        self.run_multiple_scenarios(timesteps,
                                    scenario_list=list(self.scenario_id_dict.keys())[:5])

        if save_output:
            self.dfplot_multi.to_csv('../data/all_scenarios.csv', index=False)
            self.df_person_data = pd.DataFrame(columns=['person_id', 'age', 'asymptomatic'])
            self.df_person_data['person_id'] = range(self.number_of_people)
            self.df_person_data['age'] = self.ages
            self.df_person_data['asymptomatic'] = self.is_symptomatic
            self.df_person_data.to_csv('../data/person_data.csv', index=False)

    def load_all_scenarios(self):
        self.dfplot_multi = pd.read_csv('../data/all_scenarios.csv')
        self.df_person_data = pd.read_csv('../data/person_data.csv')
        self.ages = self.df_person_data['age']
        self.is_symptomatic = self.df_person_data['asymptomatic']

    def load_a_scenario(self, scenario_id):
        self.dfplot = pd.read_csv(f'../data/all_scenarios/scenario_{scenario_id}.csv')


    def continue_simulation(self, timesteps = 100):

        for t in range(timesteps):
            self.x = self.dfplot.loc[self.dfplot['day_ind'] == t, 'x'].values
            self.y = self.dfplot.loc[self.dfplot['day_ind'] == t, 'x'].values
            self._spread_infections()
            self._update_days_infected()
            self._cure_people()

            df_ = pd.DataFrame(columns=['x', 'y', 'color', 'day', 'size', 'infected', 'person_id', 'immune'])
            df_['x'] = self.x
            df_['y'] = self.y
            df_['color'] = [self.person_colors[inf_] for inf_ in self.infected]
            df_['day_ind'] = t
            df_['size'] = 2
            df_['person_id'] = range(self.number_of_people)
            df_['infected'] = self.infected
            df_['infection_status'] = [self.infection_dict[inf_] for inf_ in self.infected]
            df_['immune'] = self.immune
            df_['deceased'] = self.deceased
            df_.loc[df_['immune'] == 1, 'infected'] = 2
            df_.loc[df_['deceased'] == 1, 'infected'] = 3
            self.dfplot = pd.concat((self.dfplot, df_), sort=False)

    def visualise_simulation(self,
                             scenario=None,
                             save_as_html=False,
                             figsize=(750, 750)):

        if scenario is not None:
            self.dfplot_multi['day'] = [f"{int(np.floor(d / self.hours_in_a_day))} {int(24 * (d % self.hours_in_a_day) / self.hours_in_a_day)}:00" for d in self.dfplot_multi['day_ind']]
            print('visualising...')
            scenario_id = self.scenario_id_dict[scenario]
            fig = px.scatter(self.dfplot_multi,
                             x=f"x_{scenario_id}",
                             y=f"y_{scenario_id}",
                             animation_frame="day",
                             size='size',
                             animation_group="person_id",
                             color=f"infected_{scenario_id}",
                             hover_name='person_id',
                             size_max=7,
                             range_x=[0, 1],
                             range_y=[0, 1],
                             range_color=[0, 3],
                             color_continuous_scale=['steelblue', 'sandybrown', 'forestgreen', 'firebrick'],
                             )
        else:
            self.dfplot['size'] = 7
            self.dfplot['day'] = [f"{int(np.floor(d / self.hours_in_a_day))} {int(24 * (d % self.hours_in_a_day) / self.hours_in_a_day)}:00" for d in self.dfplot['day_ind']]
            self.dfplot.loc[self.dfplot['day'].str.contains('0:00'), 'day'] = \
                self.dfplot.loc[self.dfplot['day'].str.contains('0:00'), 'day'].apply(lambda x: x.replace(' 0:', '   0:') + 'am')
            self.dfplot.loc[self.dfplot['day'].str.contains('12:00'), 'day'] = \
                self.dfplot.loc[self.dfplot['day'].str.contains('12:00'), 'day'].apply(lambda x: x + 'pm')
            fig = px.scatter(self.dfplot.loc[(self.dfplot['day'].str.contains(' 0'))
                             | (self.dfplot['day'].str.contains(' 12'))],
                             x="x",
                             y="y",
                             animation_frame="day",
                             size='size',
                             animation_group="person_id",
                             color="infected",
                             size_max=7,
                             range_x=[0, 1],
                             range_y=[0, 1],
                             range_color=[0, 3],
                             color_continuous_scale=['steelblue', 'sandybrown', 'forestgreen', 'firebrick'],
            )


        fig.update_layout(coloraxis={'showscale': False},
                          #config={'displayModeBar': False}
                          )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["easing"] = 'linear'
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = False


        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                showline=False,
                showspikes=False,
                showticklabels=False,
                visible=False),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showspikes=False,
                showticklabels=False,
                visible=False),
            width=figsize[0],
            height=figsize[1] + 80 + 30,
            margin={'l': 0, 'r': 50, 'b': 0, 't': 35})

        fig.update_traces(hoverinfo='none',
                          hovertext='skip',
                          hovertemplate=None,
                          hoveron='fills')

        if save_as_html:
            print(f"Saving viz for scenario {self.scenario}...")
            fig.write_html(f"../data/all_viz_files/viz_{self.scenario}.html", auto_play=False)

        return fig

    def plot_results(self, scenario=None, save_as_html=False):

        if scenario is not None:
            scenario_id = self.scenario_id_dict[scenario]
            dfarea = self.dfplot_multi[['day_ind', f'infected_{scenario_id}', 'person_id']].rename(
                columns = {f'infected_{scenario_id}': 'infected'}).groupby(
                ['day_ind', 'infected'], as_index=False)['person_id'].count()
        else:
            dfarea = self.dfplot.groupby(['day_ind', 'infected'], as_index=False)['person_id'].count()

        dfarea['person_id'] *= (100 / self.number_of_people)

        fig = go.Figure(layout={'plot_bgcolor': 'rgba(0,0,0,0)',
                                 'paper_bgcolor': 'rgba(0,0,0,0)',
                                 'width': 1400,
                                 'height': 700,
                                 'font': {'color': 'silver'},
                                 'xaxis': {'title': 'Day'},
                                 'yaxis': {'title': '% Population'}
                                })
        groups = [[1], [2], [0], [3]]
        names = ['infected', 'immune', 'not infected', 'deceased']
        colors = ['steelblue', 'sandybrown', 'forestgreen', 'firebrick']
        for gid, group in enumerate(groups):
            dfg = dfarea.loc[dfarea['infected'].isin(group)].groupby('day_ind')['person_id'].sum()
            fig.add_trace(go.Scatter(
                x=dfg.index,
                y=dfg.values,
                mode='lines',
                line=dict(width=0.5, color=colors[group[-1]]),
                stackgroup='one',
                groupnorm='percent',
                hoverinfo='text',
                hovertext=None,
                name=names[gid]
            ))

        fig.update_layout(
            showlegend=True)
        fig.update_layout(
            yaxis=dict(
                type='linear',
                range=[1, 100],
                ticksuffix='%'))

        if save_as_hmtl:
            print(f"Saving results for scenario {self.scenario}...")
            fig.write_html(f"../data/all_res_files/res_{self.scenario}.html", auto_play=False)

        return fig
    
    def matplotlib_area_plot(self, save_as_png=False):

        self.dfplot['day'] = [f"{int(np.floor(d / self.hours_in_a_day))} {int(24 * (d % self.hours_in_a_day) / self.hours_in_a_day)}:00" for d in self.dfplot['day_ind']]

        dfarea = (self.dfplot
                  .groupby(['day_ind', 'day', 'infected'], as_index=False)['person_id']
                  .count()
                  .sort_values('day_ind', ascending=True))

        f, ax = plt.subplots(1, 1, figsize=(14, 7))

        groups = [1, 2, 0, 3]
        names = ['not infected', 'infected', 'immune', 'deceased']
        colors = ['steelblue', 'sandybrown', 'forestgreen', 'firebrick']

        x = list(dfarea['day_ind'].drop_duplicates())
        x = x + [x[-(1 + i)] for i in x]

        dfarea = pd.merge(
            (dfarea.pivot(columns='infected',
                          index='day_ind',
                          values='person_id')
             .fillna(0)
             .reset_index()),
            dfarea[['day_ind', 'day']].drop_duplicates(),
            how='left',
            on='day_ind')
        for i in range(4):
            if i not in list(dfarea):
                dfarea[i] = 0
        dfarea.loc[:, range(4)] = 100 * dfarea[range(4)] / self.number_of_people

        for group_id in range(4):
            group = groups[group_id]
            if group_id == 0:
                y0 = np.zeros(self.timesteps)
            else:
                y0 = dfarea[groups[:group_id]].sum(axis=1).values
            y1 = np.fliplr(dfarea[groups[:(1 + group_id)]].sum(axis=1).values.reshape(1, -1))[0]
            y = np.concatenate([y0, y1])
            ax.fill(x, y,
                    color=colors[group],
                    alpha=0.5)

        ax.set_ylim(0, 100)
        ax.axes.grid(False)
        ax.set_xlim(0, self.timesteps + 30)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)],
                     fontsize=20)

        xticks = dfarea.loc[range(0, self.timesteps, 7 * self.hours_in_a_day), 'day_ind']
        ax.set_xticks(xticks)
        ax.set_xticklabels([1 + int(i / self.hours_in_a_day) for i in xticks])
        ax.set_xticks([])
        ax.set_xticklabels([''])
        # ax.set_xlabel('Time')

        ax.set_title('% of Population Against Time'.upper(),
                     y=1.03,
                     fontsize=20)
        # ax.text(self.timesteps/2,
        #         -10,
        #         'Time',
        #         fontsize=20)
        # ax.set_ylabel('% of Population')

        dot_offset = 20
        text_offset = 10
        ylocs = range(70, 30, -10)
        xlocs = [self.timesteps + dot_offset] * 4

        xlocs = [50, 50, 300, 300]
        ylocs = [-10, -20, -10, -20]
        self._create_final_stats_df()
        for i, yi in enumerate(ylocs):
            color = colors[i]
            ax.plot(xlocs[i],
                    yi,
                    '.',
                    markersize=20,
                    alpha=0.5)
            ax.text(x=xlocs[i] + text_offset,
                    y=yi,
                    s=f"{names[i].upper()}: {int(np.round(100*self._get_stat(names[i]) / self.number_of_people))}%",
                    verticalalignment='center',
                    horizontalalignment='left',
                    alpha=0.5,
                    color=color,
                    # fontweight='bold',
                    fontsize=20)

        if save_as_png:
            print(f"Saving matplotlib fig for scenario {self.scenario}...")
            f.savefig(f'assets/figures/matplotlib_fig_{self.scenario}.png')
        return f

    def _create_final_stats_df(self):

        self.df_final_stats = (
            self.dfplot.loc[self.dfplot['day_ind'] == self.timesteps - 1]
            .groupby('infected', as_index=False)
            ['person_id']
            .count())

        self.df_final_stats = pd.merge(pd.DataFrame(columns=['infected', 'infection_status'],
                                         data=[[0.0, 'not infected'],
                                               [1.0, 'infected'],
                                               [2.0, 'immune'],
                                               [3.0, 'deceased']]),
                            self.df_final_stats,
                            how='left',
                            on='infected').fillna(0)

    def _get_stat(self, name):
        return self.df_final_stats.loc[self.df_final_stats['infection_status'] == name, 'person_id'].values[0]