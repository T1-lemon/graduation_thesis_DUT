import TabContext from '@mui/lab/TabContext'
import TabList from '@mui/lab/TabList'
import TabPanel from '@mui/lab/TabPanel'
import Box from '@mui/material/Box'
import Tab from '@mui/material/Tab'
import * as React from 'react'

import { IMedicalExamination } from '../../../interface/MedicalExaminationInterfaces'

type Props = {
  medical?: IMedicalExamination
}

export default function TabInfomation(props: Props) {
  const { medical } = props
  const [value, setValue] = React.useState('1')

  const handleChange = (event: React.SyntheticEvent, newValue: string) => {
    setValue(newValue)
  }

  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList
            onChange={handleChange}
            aria-label='lab API tabs example'
            className='tab__list__btn'
            variant='fullWidth'
            TabIndicatorProps={{
              sx: {
                backgroundColor: '#39cabb'
              }
            }}
          >
            <Tab label='Overview' value='1' />
            <Tab label='Experience' value='2' />
          </TabList>
        </Box>
        <TabPanel value='1'>
          <div className='tab__content--item'>
            <h3>{medical?.title}</h3>
            <ul>
              {medical?.shortDescription
                .split('\n')
                .map((item: string, index: number) => {
                  return <li key={index}>{item}</li>
                })}
            </ul>
          </div>
        </TabPanel>
        <TabPanel value='2'>
          <div className='tab__content--item'>
            <h3>Treatment Of Diseases</h3>
            <ul>
              {medical?.description
                .split('\n')
                .map((item: string, index: number) => {
                  return <li key={index}>{item}</li>
                })}
            </ul>
          </div>
        </TabPanel>
      </TabContext>
    </Box>
  )
}
