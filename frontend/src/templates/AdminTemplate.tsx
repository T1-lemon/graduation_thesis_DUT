import { LocalHospital } from '@mui/icons-material'
import AccessibleIcon from '@mui/icons-material/Accessible'
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import ArrowRightIcon from '@mui/icons-material/ArrowRight'
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord'
import HandshakeIcon from '@mui/icons-material/Handshake'
import MeetingRoomIcon from '@mui/icons-material/MeetingRoom'
import TreeItem, { TreeItemProps, treeItemClasses } from '@mui/lab/TreeItem'
import TreeView from '@mui/lab/TreeView'
import Box from '@mui/material/Box'
import { SvgIconProps } from '@mui/material/SvgIcon'
import Typography from '@mui/material/Typography'
import { styled } from '@mui/material/styles'
import React, { useEffect, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'

import '../assets/css/templates/admin_template.css'
import MenuHeader from '../components/header/HeaderBoxAvatar'
import HeaderBoxLogo from '../components/header/HeaderBoxLogo'
import { settingsAdmin } from '../utils/HeaderSetting'

declare module 'react' {
  interface CSSProperties {
    '--tree-view-color'?: string
    '--tree-view-bg-color'?: string
  }
}

type StyledTreeItemProps = TreeItemProps & {
  bgColor?: string
  color?: string
  labelIcon: React.ElementType<SvgIconProps>
  labelInfo?: string
  labelText: string
}

const StyledTreeItemRoot = styled(TreeItem)(({ theme }) => ({
  color: theme.palette.text.secondary,
  [`& .${treeItemClasses.content}`]: {
    color: theme.palette.text.secondary,
    borderRadius: theme.spacing(1),
    paddingRight: theme.spacing(1),
    marginTop: '10px',
    fontWeight: theme.typography.fontWeightMedium,
    '&.Mui-expanded': {
      fontWeight: theme.typography.fontWeightRegular
    },
    '&:hover': {
      backgroundColor: 'rgba(17, 25, 39, 0.04)'
    },
    '&.Mui-focused, &.Mui-selected, &.Mui-selected.Mui-focused': {
      backgroundColor: 'rgba(17, 25, 39, 0.12)'
    },
    [`& .${treeItemClasses.label}`]: {
      fontWeight: 'inherit',
      color: 'inherit'
    },
    '&.MuiTreeItem-content': {
      position: 'relative',
      display: 'flex',
      alignItem: 'center',
      height: '35px'
    },
    '&.MuiTreeItem-content .MuiTreeItem-iconContainer': {
      position: 'absolute',
      right: '0'
    }
  },
  [`& .${treeItemClasses.group}`]: {
    marginLeft: 0,
    [`& .${treeItemClasses.content}`]: {
      paddingLeft: theme.spacing(2)
    }
  }
}))

function StyledTreeItem(props: StyledTreeItemProps) {
  const {
    bgColor,
    color,
    labelIcon: LabelIcon,
    labelInfo,
    labelText,
    ...other
  } = props

  return (
    <StyledTreeItemRoot
      label={
        <Box sx={{ display: 'flex', alignItems: 'center', p: 0.5, pr: 0 }}>
          <Box
            component={LabelIcon}
            color='inherit'
            sx={{ mr: 1 }}
            className='style__box--icon'
          />
          <Typography
            variant='body2'
            sx={{ fontWeight: 'inherit', flexGrow: 1 }}
          >
            {labelText}
          </Typography>
          <Typography variant='caption' color='inherit'>
            {labelInfo}
          </Typography>
        </Box>
      }
      style={{
        '--tree-view-color': color,
        '--tree-view-bg-color': bgColor
      }}
      {...other}
    />
  )
}

function AdminTemplate() {
  const navigate = useNavigate()
  const location = useLocation()
  const path = location.pathname
  const segments = path.split('/')
  const [url, setUrl] = useState<String>('')

  let route = segments[segments.length - 1]

  useEffect(() => {
    parseInt(route) ? setUrl(segments[segments.length - 2]) : setUrl(route)
  }, [route])

  const [scroll, setScroll] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 0) {
        setScroll(true)
      } else {
        setScroll(false)
      }
    }

    window.addEventListener('scroll', handleScroll)

    return () => {
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  return (
    <div className='admin__container'>
      <div className='admin__sidebar'>
        <div className='admin__sidebar--header admin__header'>
          <HeaderBoxLogo />
        </div>
        <div className='admin__sidebar--body'>
          <TreeView
            aria-label='gmail'
            defaultExpanded={['3']}
            defaultCollapseIcon={<ArrowDropDownIcon />}
            defaultExpandIcon={<ArrowRightIcon />}
            defaultEndIcon={<div style={{ width: 24 }} />}
            sx={{ height: 264, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}
          >
            <StyledTreeItem
              nodeId='1'
              labelText='Appointment'
              labelIcon={HandshakeIcon}
              onClick={() => {
                setUrl('appointments')
                navigate('/admin/appointments')
              }}
              className={url === 'appointments' ? 'label--focus' : ''}
            />
            <StyledTreeItem
              nodeId='2'
              labelText='Departments'
              labelIcon={MeetingRoomIcon}
              onClick={() => {
                setUrl('deparments')
                navigate('/admin/departments')
              }}
              className={url === 'departments' ? 'label--focus' : ''}
            />
            <StyledTreeItem
              nodeId='3'
              labelText='Doctors'
              labelIcon={LocalHospital}
              onClick={() => {
                url !== 'list' || url !== 'add-doctor'
                  ? setUrl('')
                  : setUrl(url)
              }}
              className={
                url === 'list' ||
                url === 'add' ||
                url === 'profile' ||
                url === 'archive'
                  ? 'label__parent--focus'
                  : ''
              }
            >
              <StyledTreeItem
                nodeId='5'
                labelText='List'
                labelIcon={FiberManualRecordIcon}
                onClick={() => {
                  setUrl('list')
                  navigate('/admin/doctors/list')
                }}
                className={`item__child ${
                  url === 'list' || url === 'profile'
                    ? 'item__child--focus'
                    : ''
                }`}
              />
              <StyledTreeItem
                nodeId='6'
                labelText='Archive'
                labelIcon={FiberManualRecordIcon}
                onClick={() => {
                  setUrl('archive')
                  navigate('/admin/doctors/archive')
                }}
                className={`item__child ${
                  url === 'archive' ? 'item__child--focus' : ''
                }`}
              />
              <StyledTreeItem
                nodeId='7'
                labelText='Add Doctor'
                labelIcon={FiberManualRecordIcon}
                onClick={() => {
                  setUrl('add')
                  navigate('/admin/doctors/add')
                }}
                className={`item__child ${
                  url === 'add' ? 'item__child--focus' : ''
                }`}
              />
              {/* <StyledTreeItem
                nodeId='8'
                labelText='Profile'
                labelIcon={FiberManualRecordIcon}
                onClick={() => {
                  setUrl('profile')
                  navigate('/admin/doctors/profile/1')
                }}
                className={`item__child ${
                  url === 'profile' ? 'item__child--focus' : ''
                }`}
              /> */}
            </StyledTreeItem>
            <StyledTreeItem
              nodeId='9'
              labelText='Patients'
              labelIcon={AccessibleIcon}
              onClick={() => {
                setUrl('patients')
                navigate('/admin/patients')
              }}
              className={url === 'patients' ? 'label--focus' : ''}
            />
          </TreeView>
        </div>
      </div>
      <div className='admin__content--container'>
        <div
          className={`admin__header admin__content--header ${
            scroll ? 'admin__content--header-scrolled' : ''
          }`}
        >
          <MenuHeader settings={settingsAdmin} />
        </div>
        <div className='admin__outlet--container'>
          <Outlet />
        </div>
      </div>
    </div>
  )
}

export default AdminTemplate
