import s from './Sidebar.module.scss';
import React, { createContext, useContext } from 'react';
import { PhaseTimeline } from './PhaseTimeline';
import { Commentary } from './Commentary';
import { IProgramState } from './Program';
import { useSubscriptions } from '../utils/hooks';

export const WalkthroughSidebar: React.FC = () => {
    return <div className={s.walkthrough}>
        <div className={s.split}>

            <div className={s.timelineLeft}>
                <PhaseTimeline />
            </div>

            <div className={s.content}>
                <Commentary />
            </div>

        </div>
    </div>;
};

export let ProgramStateContext = createContext<IProgramState>(null!);

export function useProgramState() {
    let context = useContext(ProgramStateContext);
    useSubscriptions(context?.htmlSubs);
    return context;
}
